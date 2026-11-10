from typing import Callable, Literal, List
import os
import sys
import json
import gzip
import zstandard
from pathlib import Path

from PIL import ImageFont

from mpixel.renderer.processor import _text_to_units
from mpixel.renderer.functional import _measure_single_unit, _normalize_pad


UnitType = Literal['words', 'chars', '4-grams']
CompressionType = Literal['none', 'gzip', 'zstd']


LANGUAGES: List[str] = [
    'eng_Latn',  # English -- not a part of FineWeb-2
    'ukr_Cyrl',  # Ukrainian
    'hin_Deva',  # Hindi
    'cmn_Hani',  # Mandarin Chinese
    'spa_Latn',  # Spanish
    'fin_Latn',  # Finnish
    'tur_Latn',  # Turkish
    'fas_Arab',  # Persian
    'arb_Arab',  # Standard Arabic
    'heb_Hebr',  # Hebrew
    'kor_Hang',  # Korean
    'jpn_Jpan',  # Japanese
    'tel_Telu',  # Telugu
    'swh_Latn',  # Swahili
    'tha_Thai',  # Thai
    'ell_Grek',  # Modern Greek
    'kat_Geor',  # Georgian
    'vie_Latn',  # Vietnamese
    'ind_Latn',  # Indonesian
    'kaz_Cyrl',  # Kazakh
    'khk_Cyrl',  # Halh Mongolian
    'dan_Latn',  # Danish
]


DEFAULT_FONT = "NotoSans-Regular.ttf"
SCRIPT_TO_FILENAME_DICT = {
    "Latn": "NotoSans-Regular.ttf",
    "Cyrl": "NotoSans-Regular.ttf",
    "Grek": "NotoSans-Regular.ttf",
    "Arab": "NotoSansArabic-Regular.ttf",
    "Hebr": "NotoSansHebrew-Regular.ttf",
    "Deva": "NotoSansDevanagari-Regular.ttf",
    "Telu": "NotoSansTelugu-Regular.ttf",
    "Thai": "NotoSansThai-Regular.ttf",
    "Geor": "NotoSansGeorgian-Regular.ttf",
    "Hang": "NotoSansKR-Regular.ttf",
    "Jpan": "NotoSansJP-Regular.ttf",
    "Hani": "NotoSansCJKsc-Regular.otf",
}


def get_unit(language: str) -> UnitType:
    assert language in LANGUAGES
    if language in ('cmn_Hani', 'jpn_Jpan', 'kor_Hang'):
        return 'chars'
    if language == 'tha_Thai':
        return '4-grams'
    return 'words'


def count_units(text: str, unit: UnitType) -> int:
    if unit == 'words':
        return len(text.split())
    elif unit == 'chars':
        return len(text)
    elif unit == '4-grams':
        return len(text) // 4
    else:
        raise ValueError(f'Unknown unit type: {unit}')


def create_compute_text_width_fn(language: str, font_dir: str, font_size: int = 16) -> Callable:
    """
    This higher-order function takes input arguments of `language`, `font_dir`, and `font_size`,
        and returns a closure which computes the width of the given input text.
    """
    assert language in LANGUAGES
    script = language.split("_")[1]

    # Load the font
    font_name = SCRIPT_TO_FILENAME_DICT.get(script, DEFAULT_FONT)
    font_path = os.path.join(font_dir, font_name)
    font = ImageFont.truetype(font_path, font_size)

    # The closure function
    def measure(text: str) -> int:
        return int(font.getlength(text))

    return measure


def create_measure_unit_patches_fn(
    block_size: int = 16,
    font_size: int = 12,
    pad: int | tuple[int, int] = 1,
) -> Callable[[str], int]:
    pad_left, pad_right = _normalize_pad(pad)

    def measure(unit_text: str) -> int:
        unit_width, _, _ = _measure_single_unit(
            unit_text, block_size, font_size, pad_left, pad_right
        )
        return unit_width // block_size

    return measure


def _join_units(units: List[str], segment_starts: List[int]) -> str:
    parts: List[str] = []
    for i, unit in enumerate(units):
        if i > 0 and segment_starts[i]:
            parts.append(" ")
        parts.append(unit)
    return "".join(parts)


def split_text_by_patches(
    text: str,
    measure_fn: Callable[[str], int],
    max_patches: int,
    strategy: str = "auto",
    trim_partial_words: bool = False,
) -> List[dict]:
    units, unit_segment_starts = _text_to_units(text, strategy)
    if not units:
        return []

    chunks: List[dict] = []
    i = 0
    start = 0
    cur_patches = 0
    cur_words = 0
    cur_chars = 0

    while i < len(units):
        unit = units[i]
        p = measure_fn(unit)
        if cur_patches + p > max_patches and i > start:
            chunk_text = _join_units(units[start:i], unit_segment_starts[start:i])
            chunks.append({
                'text': chunk_text,
                'num_patches': cur_patches,
                'num_words': cur_words,
                'num_chars': cur_chars,
            })
            if trim_partial_words:
                while i < len(units) and not unit_segment_starts[i]:
                    i += 1
                if i >= len(units):
                    break
            start = i
            cur_patches = 0
            cur_words = 0
            cur_chars = 0
            continue

        cur_patches += p
        cur_chars += len(unit)
        if unit_segment_starts[i]:
            cur_words += 1
        i += 1

    if start < len(units):
        chunk_text = _join_units(units[start:], unit_segment_starts[start:])
        chunks.append({
            'text': chunk_text,
            'num_patches': cur_patches,
            'num_words': cur_words,
            'num_chars': cur_chars,
        })

    return chunks


def split_text_by_width_binary(
    text: str,
    measure: Callable[[str], int],
    max_width: int,
    unit: str,
) -> List[str]:
    """
    Split `text` into chunks such that each chunk's pixel width <= max_width,
    using a binary-search strategy on token indices.
    """

    assert unit in ("words", "chars", "4-grams")
    def _tokenize_by_unit(text: str, unit: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if unit == "words":
            return text.split()
        elif unit == "chars":
            return list(text)
        elif unit == "4-grams":
            return [text[i : i + 4] for i in range(0, len(text), 4)]

    def _join_tokens(tokens: List[str], unit: str) -> str:
        if unit == "words":
            return " ".join(tokens)
        else:
            return "".join(tokens)

    tokens = _tokenize_by_unit(text, unit)
    if not tokens:
        return []

    full_text = _join_tokens(tokens, unit)
    if measure(full_text) <= max_width:
        return [full_text]

    chunks: List[str] = []
    start = 0

    while start < len(tokens):
        remaining = tokens[start:]
        remaining_text = _join_tokens(remaining, unit)

        # Terminate if the remaining part fits.
        if measure(remaining_text) <= max_width:
            chunks.append(remaining_text)
            break

        # Binary search for the largest prefix that fits.
        lo, hi = 1, len(remaining)
        best = 1  # fall back to at least one token

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_tokens = remaining[:mid]
            candidate_text = _join_tokens(candidate_tokens, unit)
            w = measure(candidate_text)

            if w <= max_width:
                best = mid
                lo = mid + 1      # try a larger prefix
            else:
                hi = mid - 1      # too wide, try smaller

        # Emit the best-fitting prefix
        chunk_tokens = remaining[:best]
        chunk_text = _join_tokens(chunk_tokens, unit)
        chunks.append(chunk_text)

        # Move start past the emitted tokens
        start += best

    return chunks


def split_text_by_width_fast(
    text: str,
    measure: Callable[[str], int],
    max_width: int,
    unit: str,
) -> List[str]:
    """
    Split `text` into chunks such that each chunk's pixel width <= max_width.

    This uses a proportional approximation strategy:
    1. Measure the full text width ONCE
    2. For each chunk: calculate proportion based on estimated remaining_width
    3. Allocate tokens proportionally without re-measuring
    4. Update remaining_width by subtracting max_width

    Note: Chunks may exceed max_width slightly; truncation happens in later phases.
    """
    assert unit in ("words", "chars", "4-grams")

    def _tokenize_by_unit(text: str, unit: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if unit == "words":
            return text.split()
        elif unit == "chars":
            return list(text)
        elif unit == "4-grams":
            return [text[i : i + 4] for i in range(0, len(text), 4)]

    def _join_tokens(tokens: List[str], unit: str) -> str:
        if unit == "words":
            return " ".join(tokens)
        else:
            return "".join(tokens)

    tokens = _tokenize_by_unit(text, unit)
    if not tokens:
        return []

    full_text = _join_tokens(tokens, unit)
    full_width = measure(full_text)

    # Quick check: if entire text fits, return immediately
    if full_width <= max_width:
        return [full_text]

    chunks: List[str] = []
    start = 0
    remaining_width = full_width

    while start < len(tokens):
        remaining = tokens[start:]
        num_remaining = len(remaining)

        if num_remaining == 0:
            break

        # Calculate proportion and estimate how many tokens will fit
        proportion = max_width / remaining_width
        estimated_tokens = max(1, int(num_remaining * proportion))

        # Build chunk with estimated tokens (no verification!)
        chunk_tokens = remaining[:estimated_tokens]
        chunk_text = _join_tokens(chunk_tokens, unit)
        chunks.append(chunk_text)

        start += estimated_tokens

        # Update remaining_width by subtracting max_width (proportional approximation)
        remaining_width -= max_width

    return chunks


class _ZstdTextWrapper:
    def __init__(self, stream_writer, raw_file):
        self._writer = stream_writer
        self._raw = raw_file

    def write(self, text: str):
        self._writer.write(text.encode("utf-8"))

    def close(self):
        self._writer.close()
        self._raw.close()


def open_output_file(
    output_dir: Path,
    file_prefix: str,
    file_index: int,
    compression: CompressionType,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if compression == "gzip":
        ext = ".jsonl.gz"
    elif compression == "zstd":
        ext = ".jsonl.zst"
    else:
        ext = ".jsonl"
    path = output_dir / f"{file_prefix}-{file_index:08d}{ext}"

    if compression == "gzip":
        f = gzip.open(path, "wt", encoding="utf-8")
    elif compression == "zstd":
        cctx = zstandard.ZstdCompressor(level=3)
        raw = path.open("wb")
        f = cctx.stream_writer(raw)
        f = _ZstdTextWrapper(f, raw)
    else:
        f = path.open("w", encoding="utf-8")

    print(f"[info] Opening new shard: {path}", file=sys.stderr)
    return f, path


def _stream_to_shards(
    iterator,
    *,
    language: str,
    max_patches: int,
    output_dir: Path,
    file_prefix: str,
    patches_per_file: int,
    compression: CompressionType,
    measure_fn: Callable[[str], int],
    num_patches: int,
    text_column: str,
    split_name: str,
    metadata: dict,
    strategy: str = "auto",
) -> tuple:
    """Stream examples from an iterator into sharded JSONL files.

    Each document is split into chunks using adaptive rendering units
    (bigrams, monograms, or words depending on script). Chunks are
    limited by ``num_patches`` (max patches per sequence).

    Args:
        iterator: Iterator over dataset examples (consumed in-place).
        max_patches: Stop after this many total patches. 0 means no limit.
        patches_per_file: Rotate to a new shard after this many patches.
        num_patches: Max patches per sequence (chunk).
        split_name: Name of the split (for logging).
        metadata: Base metadata dict. Will be updated with counts
            and written to a JSON file.

    Returns:
        (total_patches, total_words, total_chars, num_examples) streamed.
    """
    total_patches = 0
    total_words = 0
    total_chars = 0
    num_examples = 0
    file_patches = 0
    file_index = 0

    out_f, current_path = open_output_file(
        output_dir, file_prefix, file_index, compression
    )

    try:
        for example in iterator:
            text = example.get(text_column)
            if not text:
                continue

            chunks = split_text_by_patches(
                text=text,
                measure_fn=measure_fn,
                max_patches=num_patches,
                strategy=strategy,
            )
            if not chunks:
                continue

            num_examples += 1

            for chunk_id, chunk in enumerate(chunks):
                record = {
                    "language": language,
                    "text": chunk['text'],
                    "chunk-id": chunk_id,
                    "num_patches": chunk['num_patches'],
                    "num_words": chunk['num_words'],
                    "num_chars": chunk['num_chars'],
                }
                if "id" in example:
                    record["id"] = example["id"]

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_patches += chunk['num_patches']
                total_words += chunk['num_words']
                total_chars += chunk['num_chars']
                file_patches += chunk['num_patches']

            if num_examples % 1000 == 0:
                print(
                    f"[{split_name}] docs={num_examples}, "
                    f"total_patches={total_patches}, "
                    f"total_words={total_words}, "
                    f"total_chars={total_chars}, "
                    f"shard={file_index}, "
                    f"file_patches={file_patches}",
                    file=sys.stderr,
                )

            if max_patches > 0 and total_patches >= max_patches:
                print(f"[{split_name}] Reached threshold: {total_patches} patches", file=sys.stderr)
                break

            if file_patches >= patches_per_file:
                out_f.close()
                file_index += 1
                file_patches = 0
                out_f, current_path = open_output_file(
                    output_dir, file_prefix, file_index, compression
                )
    finally:
        try:
            out_f.close()
        except Exception:
            pass

    num_shards = file_index + 1

    print(
        f"[{split_name}] Finished: docs={num_examples}, "
        f"patches={total_patches}, words={total_words}, "
        f"chars={total_chars}, shards={num_shards}",
        file=sys.stderr,
    )

    metadata.update({
        "num_examples": num_examples,
        "total_patches": total_patches,
        "total_words": total_words,
        "total_chars": total_chars,
        "num_shards": num_shards,
    })
    metadata_path = output_dir / f"{file_prefix}-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return total_patches, total_words, total_chars, num_examples
