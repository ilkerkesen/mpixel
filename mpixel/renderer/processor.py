import unicodedata
from functools import lru_cache
from itertools import groupby
from typing import Any

import numpy as np
import regex
from font_download import FontConfig
from pixel_renderer import PixelRendererProcessor

from mpixel.renderer.functional import _render_units


_GRAPHEME_RE = regex.compile(r"\X")

# ZWSP is a zero-width word boundary but ``str.isspace()`` returns False for
# it, so we treat it as whitespace explicitly. Other format chars (ZWJ, ZWNJ,
# WJ) are kept because they participate in grapheme clusters.
_EXTRA_WHITESPACE = frozenset({"\u200b"})

# Scripts that must be rendered word-at-a-time. Connected-cursive scripts
# (Arabic, N'Ko, Syriac, …) need this because letter shape depends on
# neighbours; Hebrew is non-cursive but benefits from whole-word rendering
# to keep diacritics and final forms in proper context.
_WORD_SCRIPT_RE = regex.compile(
    r"[\p{sc=Arabic}\p{sc=Hebrew}\p{sc=Nko}\p{sc=Syriac}\p{sc=Adlam}"
    r"\p{sc=Mongolian}\p{sc=Mandaic}\p{sc=Hanifi_Rohingya}]"
)

# Scripts where a single grapheme cluster is roughly em-wide (≈ one patch).
# BIGRAMS here degenerates into 2-patch units with mid-character slicing;
# n=1 gives one clean character per patch at the same sequence length.
_EM_WIDTH_RE = regex.compile(
    r"[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Hangul}]"
)


@lru_cache(maxsize=32768)
def _auto_strategy_for_segment(segment: str) -> tuple[str, int]:
    """Pick per-segment ``(strategy, n)`` for ``strategy="auto"``.

    Priority: word-scripts > em-width > bigrams. Word-scripts win when both
    present because shape correctness dominates unit sizing — a segment mixing
    Arabic and CJK (rare) must preserve Arabic joining, and CJK-as-a-unit is
    still a correct rendering.
    """
    if _WORD_SCRIPT_RE.search(segment):
        return "words", 1  # n ignored for words
    if _EM_WIDTH_RE.search(segment):
        return "ngrams", 1
    return "ngrams", 2


@lru_cache(maxsize=32768)
def _words_strategy_for_segment(segment: str) -> tuple[str, int]:
    """Pick per-segment ``(strategy, n)`` for ``strategy="words"``.

    CJK scripts (Han, Hiragana, Katakana, Hangul) → monogram (one grapheme
    per patch); every other script → whole-word rendering.
    """
    if _EM_WIDTH_RE.search(segment):
        return "ngrams", 1
    return "words", 1  # n ignored for words


def _build_segment_start_mask(
    unit_segment_starts: list[int],
    patch_counts: list[int],
) -> np.ndarray:
    flags = np.array(unit_segment_starts, dtype=np.int64)
    counts = np.array(patch_counts)
    keep = counts > 0
    flags = flags[keep]
    counts = counts[keep]
    total = int(counts.sum())
    mask = np.zeros(total, dtype=np.int64)
    if total > 0:
        starts = np.zeros(len(counts), dtype=np.intp)
        np.cumsum(counts[:-1], out=starts[1:])
        mask[starts] = flags
    return mask


def _is_whitespace(grapheme: str) -> bool:
    # Any whitespace (or ZWSP) character inside the cluster is enough to drop
    # it — handles defective combining sequences like " \u0301" where the
    # cluster as a whole isn't str.isspace().
    return any(ch.isspace() or ch in _EXTRA_WHITESPACE for ch in grapheme)


@lru_cache(maxsize=32768)
def _first_strong_direction(segment: str) -> str:
    """Return ``"RTL"`` if the first strong bidi character in ``segment`` is
    ``R`` or ``AL``, else ``"LTR"``. Segments with no strong bidi character
    (pure punctuation, digits) default to LTR.
    """
    for ch in segment:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL"):
            return "RTL"
        if bidi == "L":
            return "LTR"
    return "LTR"


def _chunk_ngrams(segment: str, n: int) -> list[str]:
    graphemes = _GRAPHEME_RE.findall(segment)
    return ["".join(graphemes[i:i + n]) for i in range(0, len(graphemes), n)]


def _chunk_segment(segment: str, strategy: str, n: int) -> list[str]:
    if strategy == "words":
        return [segment]
    return _chunk_ngrams(segment, n)


def _segment_graphemes(graphemes: list[str]) -> list[str]:
    """Split a grapheme list on whitespace. Each maximal run of non-whitespace
    graphemes is one segment; runs of whitespace graphemes survive here as
    whitespace-only segments so they break neighbouring runs, and the caller
    drops them before rendering. Punctuation, apostrophes, ellipsis, dashes,
    and emoji are absorbed into whichever whitespace-delimited chunk they
    happen to sit in — segments are "visual words", not class-homogeneous.
    """
    return [
        "".join(group)
        for _, group in groupby(graphemes, key=_is_whitespace)
    ]


def _segments_to_units(
    text: str,
    strategy: str = "bigrams",
    n: int = 2,
) -> tuple[list[str], list[int]]:
    """Split ``text`` into segments, apply lean RTL reordering, chunk each
    segment according to ``strategy``, and return the units with a parallel
    start-flag array.

    ``strategy`` is one of:

    - ``"words"``: each whitespace-delimited segment is one unit.
    - ``"bigrams"``: grapheme pairs within each segment (equivalent to
      ``strategy="ngrams", n=2``).
    - ``"ngrams"``: grapheme ``n``-grams within each segment.
    - ``"auto"``: per-segment dispatch based on script. Word-level scripts
      (Arabic, Hebrew, N'Ko, Syriac, Adlam, Mongolian, Mandaic, Hanifi
      Rohingya) use ``"words"`` to preserve shaping context; em-width
      scripts (Han, Hiragana, Katakana, Hangul) use ``n=1`` because a
      single character already fills one patch; everything else falls
      back to bigrams.

    ``unit_segment_starts[i] == 1`` iff unit ``i`` is the first unit of a
    segment. The first segment of the text always gets ``1``.
    """
    graphemes = _GRAPHEME_RE.findall(text)
    segments = _segment_graphemes(graphemes)
    tagged = [
        (seg, _first_strong_direction(seg))
        for seg in segments
        if not _is_whitespace(seg)
    ]

    # Group adjacent same-direction segments into runs; reverse the segment
    # order of each RTL run. Graphemes inside a chunk are left untouched —
    # Pango applies within-chunk BiDi at render time.
    ordered: list[tuple[str, str]] = []
    for direction, group in groupby(tagged, key=lambda p: p[1]):
        run = list(group)
        if direction == "RTL":
            run.reverse()
        ordered.extend(run)

    units: list[str] = []
    unit_segment_starts: list[int] = []
    for segment, direction in ordered:
        if strategy == "auto":
            seg_strategy, seg_n = _auto_strategy_for_segment(segment)
        elif strategy == "words":
            seg_strategy, seg_n = _words_strategy_for_segment(segment)
        else:
            seg_strategy, seg_n = strategy, n
        chunks = _chunk_segment(segment, seg_strategy, seg_n)
        if direction == "RTL":
            chunks.reverse()
        for i, chunk in enumerate(chunks):
            units.append(chunk)
            unit_segment_starts.append(1 if i == 0 else 0)

    return units, unit_segment_starts


def _text_to_units(
    text: str,
    strategy: str = "auto",
    n: int = 2,
) -> tuple[list[str], list[int]]:
    """Segment text into rendering units in logical (memory) order.

    Same adaptive chunking as :func:`_segments_to_units` but without RTL
    reordering, so units stay in the original text order. This makes
    rendered output consistent regardless of script direction and avoids
    double-reversal when text is stored and later re-rendered.
    """
    graphemes = _GRAPHEME_RE.findall(text)
    segments = _segment_graphemes(graphemes)

    units: list[str] = []
    unit_segment_starts: list[int] = []
    for seg in segments:
        if _is_whitespace(seg):
            continue
        if strategy == "auto":
            seg_strategy, seg_n = _auto_strategy_for_segment(seg)
        elif strategy == "words":
            seg_strategy, seg_n = _words_strategy_for_segment(seg)
        else:
            seg_strategy, seg_n = strategy, n
        chunks = _chunk_segment(seg, seg_strategy, seg_n)
        for i, chunk in enumerate(chunks):
            units.append(chunk)
            unit_segment_starts.append(1 if i == 0 else 0)

    return units, unit_segment_starts


class BigramsRenderingProcessor:
    """HuggingFace-style processor that turns raw text into a rendered strip
    plus per-patch alignment metadata.

    The ``strategy`` argument selects how text is segmented into rendering
    units before block-aligned rasterisation:

    - ``"bigrams"``: split text on whitespace — each maximal run of
      non-whitespace graphemes is one segment ("visual word"). Punctuation,
      apostrophes, dashes, ellipsis, and emoji are absorbed into the adjacent
      chunk they sit in, so ``"Hello,"``, ``"can't"``, ``"3.14"``,
      ``"state-of-the-art"``, ``"wait..."``, and ``"(Hello)"`` are each one
      segment. Within each segment, grapheme clusters are chunked into pairs;
      a final odd grapheme becomes a 1-grapheme unit.

    **Whitespace contract.** Every ``str.isspace()`` character is dropped
    before rendering; inter-segment boundaries are carried entirely by
    ``segment_start_mask``. Consequence: ``"a b"``, ``"a\\tb"``, ``"a\\nb"``,
    and ``"a\\u00a0b"`` all produce the same units, pixels, and mask.

    **RTL.** Lean visual ordering at the segment level, not full Unicode BiDi.
    Segments are classified LTR or RTL by their first strong bidi character;
    adjacent same-direction segments are grouped into runs; RTL runs have
    both their segment order and their per-segment bigram order reversed.
    Graphemes inside a bigram are unchanged — Pango handles within-bigram
    BiDi at render time. Segments with no strong bidi character default to
    LTR.

    Returned ``segment_start_mask`` is a length-``N_patches`` int array with
    ``1`` on the first patch of each segment and ``0`` elsewhere. The first
    segment of the text always gets ``1``.

    **Font / reproducibility.** The renderer uses the Pango ``"sans"`` family,
    which fontconfig resolves against the installed fonts of the current
    process. Pass a ``FontConfig`` (e.g. ``FontConfig(sources=FONTS_NOTO_SANS)``)
    to pin the font set so patch widths, glyphs, and mask lengths are
    independent of the host environment. With ``font=None`` the processor
    falls back to system fonts and reproducibility is *not* guaranteed.
    """

    def __init__(self, strategy: str = "bigrams", font: FontConfig | None = None):
        if strategy != "bigrams":
            raise NotImplementedError(f"strategy={strategy!r} is not implemented yet")
        self.strategy = strategy
        self.font = font
        self._font_owner = PixelRendererProcessor(font=font) if font is not None else None

    def __call__(
        self,
        text: str,
        block_size: int = 16,
        font_size: int = 12,
        pad: int | tuple[int, int] = 1,
    ) -> dict[str, Any]:
        if self._font_owner is not None:
            self._font_owner._ensure_fontconfig_initialized()

        units, unit_segment_starts = _text_to_units(text, self.strategy)
        image, patch_counts = _render_units(
            units, block_size=block_size, font_size=font_size, pad=pad
        )

        segment_start_mask = _build_segment_start_mask(unit_segment_starts, patch_counts)

        sep_patch = np.zeros((block_size, block_size, 3), dtype=np.uint8)
        image = np.concatenate([image, sep_patch], axis=1)
        segment_start_mask = np.append(segment_start_mask, 0)

        return {
            "pixel_values": image,
            "segment_start_mask": segment_start_mask,
        }


class AdaptiveRenderingProcessor:
    """HuggingFace-style processor that renders text with script-aware
    segmentation.

    Each whitespace-delimited segment is chunked according to its script:
    word-level scripts (Arabic, Hebrew, N'Ko, Syriac, Adlam, Mongolian,
    Mandaic, Hanifi Rohingya) render as whole words so Pango can resolve
    contextual shaping; em-width scripts (Han, Hiragana, Katakana, Hangul)
    render one grapheme per patch; all other scripts fall back to bigrams.

    The whitespace and RTL semantics match :class:`BigramsRenderingProcessor`.

    See :class:`BigramsRenderingProcessor` for the font-reproducibility contract.
    """

    def __init__(self, font: FontConfig | None = None):
        self.font = font
        self._font_owner = PixelRendererProcessor(font=font) if font is not None else None

    def __call__(
        self,
        text: str,
        block_size: int = 16,
        font_size: int = 12,
        pad: int | tuple[int, int] = 1,
    ) -> dict[str, Any]:
        if self._font_owner is not None:
            self._font_owner._ensure_fontconfig_initialized()

        units, unit_segment_starts = _text_to_units(text, "auto")
        image, patch_counts = _render_units(
            units, block_size=block_size, font_size=font_size, pad=pad
        )

        segment_start_mask = _build_segment_start_mask(unit_segment_starts, patch_counts)

        sep_patch = np.zeros((block_size, block_size, 3), dtype=np.uint8)
        image = np.concatenate([image, sep_patch], axis=1)
        segment_start_mask = np.append(segment_start_mask, 0)

        return {
            "pixel_values": image,
            "segment_start_mask": segment_start_mask,
        }


class WordsRenderingProcessor:
    """HuggingFace-style processor that renders text with word-level
    segmentation, except for CJK scripts which use monogram rendering.

    CJK scripts (Han, Hiragana, Katakana, Hangul) render one grapheme per
    patch. Every other script — including cursive scripts like Arabic — renders
    each whitespace-delimited segment as a single unit (whole-word rendering).

    The whitespace and RTL semantics match :class:`BigramsRenderingProcessor`.

    See :class:`BigramsRenderingProcessor` for the font-reproducibility contract.
    """

    def __init__(self, font: FontConfig | None = None):
        self.font = font
        self._font_owner = PixelRendererProcessor(font=font) if font is not None else None

    def __call__(
        self,
        text: str,
        block_size: int = 16,
        font_size: int = 12,
        pad: int | tuple[int, int] = 1,
    ) -> dict[str, Any]:
        if self._font_owner is not None:
            self._font_owner._ensure_fontconfig_initialized()

        units, unit_segment_starts = _text_to_units(text, "words")
        image, patch_counts = _render_units(
            units, block_size=block_size, font_size=font_size, pad=pad
        )

        segment_start_mask = _build_segment_start_mask(unit_segment_starts, patch_counts)

        sep_patch = np.zeros((block_size, block_size, 3), dtype=np.uint8)
        image = np.concatenate([image, sep_patch], axis=1)
        segment_start_mask = np.append(segment_start_mask, 0)

        return {
            "pixel_values": image,
            "segment_start_mask": segment_start_mask,
        }
