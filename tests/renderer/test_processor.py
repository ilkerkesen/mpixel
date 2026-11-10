"""Tests for mpixel.renderer.processor — text → rendered strip + segment_start_mask."""

import numpy as np
import pytest
import regex

from mpixel.renderer.processor import (
    AdaptiveRenderingProcessor,
    BigramsRenderingProcessor,
    _auto_strategy_for_segment,
    _segments_to_units,
)


BLOCK_SIZE = 16
FONT_SIZE = 12


@pytest.fixture
def processor():
    return BigramsRenderingProcessor(strategy="bigrams")


def _patches(output):
    return output["pixel_values"].shape[1] // BLOCK_SIZE


class TestSegmentsToUnits:
    """Direct tests on the pure _segments_to_units helper."""

    # --- segmentation ---

    def test_empty(self):
        assert _segments_to_units("") == ([], [])

    def test_simple_words(self):
        units, starts = _segments_to_units("hello world")
        assert units == ["he", "ll", "o", "wo", "rl", "d"]
        assert starts == [1, 0, 0, 1, 0, 0]

    def test_first_segment_starts_with_one(self):
        _, starts = _segments_to_units("hello")
        assert starts == [1, 0, 0]

    def test_trailing_punctuation_joins_word(self):
        # "Hello," is one segment — punctuation attaches to the adjacent
        # whitespace-delimited chunk, not its own segment.
        units, starts = _segments_to_units("Hello,")
        assert units == ["He", "ll", "o,"]
        assert starts == [1, 0, 0]

    def test_leading_punctuation_joins_following_word(self):
        # "'tis" has no whitespace, so it's one segment "'tis".
        units, starts = _segments_to_units("'tis")
        assert units == ["'t", "is"]
        assert starts == [1, 0]

    def test_parentheses_join_enclosed_word(self):
        units, starts = _segments_to_units("(Hello)")
        assert units == ["(H", "el", "lo", ")"]
        assert starts == [1, 0, 0, 0]

    def test_interior_apostrophe_glues_word(self):
        # "can't", "don't" (curly or ASCII) → each is one segment.
        for text, expected in [
            ("can't", ["ca", "n'", "t"]),
            ("don't", ["do", "n'", "t"]),
            ("don\u2019t", ["do", "n\u2019", "t"]),
            ("boys'", ["bo", "ys", "'"]),
        ]:
            units, starts = _segments_to_units(text)
            assert units == expected, text
            assert starts == [1] + [0] * (len(expected) - 1), text

    def test_ellipsis_joins_preceding_word(self):
        units, starts = _segments_to_units("wait...")
        assert units == ["wa", "it", "..", "."]
        assert starts == [1, 0, 0, 0]

    def test_repeated_exclamation_joins_word(self):
        units, starts = _segments_to_units("Wait!!!")
        assert units == ["Wa", "it", "!!", "!"]
        assert starts == [1, 0, 0, 0]

    def test_isolated_punctuation_run_is_one_segment(self):
        # No surrounding non-whitespace → the punctuation run is itself the
        # "visual word".
        for text, expected in [("?!", ["?!"]), ("---", ["--", "-"])]:
            units, starts = _segments_to_units(text)
            assert units == expected, text
            assert starts == [1] + [0] * (len(expected) - 1), text

    def test_whitespace_separates_punctuation_runs(self):
        units, starts = _segments_to_units("!! ??")
        assert units == ["!!", "??"]
        assert starts == [1, 1]

    def test_number_with_interior_dot_is_one_segment(self):
        units, starts = _segments_to_units("3.14")
        assert units == ["3.", "14"]
        assert starts == [1, 0]

    def test_hyphenated_compound_is_one_segment(self):
        units, starts = _segments_to_units("state-of-the-art")
        assert starts == [1] + [0] * (len(units) - 1)

    def test_thai_no_whitespace_single_segment(self):
        units, starts = _segments_to_units("สวัสดีครับ")
        assert starts[0] == 1
        assert starts[1:] == [0] * (len(units) - 1)

    # --- whitespace ---

    def test_all_whitespace_types_are_equivalent(self):
        # Any str.isspace() character acts as a segment boundary and is
        # dropped from the rendered stream — space, tab, newline, CR, NBSP
        # all produce the same units and mask.
        reference = _segments_to_units("a b")
        for text in ["a\tb", "a\nb", "a\rb", "a\u00a0b"]:
            assert _segments_to_units(text) == reference, text

    def test_consecutive_whitespace_collapses(self):
        assert _segments_to_units("a b") == _segments_to_units("a     b")
        assert _segments_to_units("a b") == _segments_to_units("a\n\n\nb")

    def test_leading_and_trailing_whitespace_ignored(self):
        assert _segments_to_units("hello") == _segments_to_units("  hello  ")

    # --- graphemes, emoji ---

    def test_combining_marks_stay_glued(self):
        # "café" written as c-a-f-e-◌́ has 4 graphemes → bigrams ["ca", "fé"].
        # The combining acute must not detach from its base 'e'.
        units, _ = _segments_to_units("cafe\u0301")
        assert units == ["ca", "fe\u0301"]
        assert regex.findall(r"\X", units[1]) == ["f", "e\u0301"]

    def test_zwj_emoji_family_stays_one_segment(self):
        family = "👨\u200d👩\u200d👧"
        units, starts = _segments_to_units(family)
        assert units == [family]
        assert starts == [1]

    def test_skin_tone_emoji_stays_one_grapheme(self):
        units, _ = _segments_to_units("👍🏽")
        assert units == ["👍🏽"]

    def test_regional_indicator_flag_stays_one_grapheme(self):
        units, _ = _segments_to_units("🇩🇰")
        assert units == ["🇩🇰"]

    def test_keycap_emoji_preserved(self):
        # "1️⃣" is a single grapheme under UAX 29 — must not be split into
        # a digit and a marks run by the segmenter.
        units, starts = _segments_to_units("1\ufe0f\u20e3")
        assert units == ["1\ufe0f\u20e3"]
        assert starts == [1]

    def test_indic_cluster_with_zwj_is_one_segment(self):
        # Devanagari virama + ZWJ + consonant must stay in a single word
        # segment even though it can span multiple grapheme clusters under
        # UAX 29.
        _, starts = _segments_to_units("क्\u200dष")
        assert starts.count(1) == 1

    def test_zwsp_is_a_word_boundary(self):
        # ZWSP (U+200B) isn't str.isspace() but acts as a zero-width word
        # boundary — "a\u200bb" must match "a b".
        assert _segments_to_units("a\u200bb") == _segments_to_units("a b")

    def test_whitespace_in_defective_grapheme_is_dropped(self):
        # "a \u0301b" — \X groups the space with the combining accent as one
        # cluster. The cluster as a whole isn't str.isspace(), but it still
        # contains a space character and must be dropped.
        assert _segments_to_units("a \u0301b") == _segments_to_units("a b")

    # --- RTL ---

    def test_pure_rtl_reverses_bigram_order(self):
        # "אבגד" → logical bigrams ["אב", "גד"] → visual ["גד", "אב"].
        units, starts = _segments_to_units("אבגד")
        assert units == ["גד", "אב"]
        assert starts == [1, 0]

    def test_rtl_odd_graphemes(self):
        # "אבג" → logical ["אב", "ג"] → visual ["ג", "אב"].
        units, starts = _segments_to_units("אבג")
        assert units == ["ג", "אב"]
        assert starts == [1, 0]

    def test_rtl_two_segments_reverse_run_order(self):
        # One RTL run of two segments → run reversed → ["גד", "אב"].
        units, starts = _segments_to_units("אב גד")
        assert units == ["גד", "אב"]
        assert starts == [1, 1]

    def test_ltr_and_rtl_runs_do_not_mix(self):
        # Each direction stays in its own run; no reordering across boundary.
        units, starts = _segments_to_units("hi אב")
        assert units == ["hi", "אב"]
        assert starts == [1, 1]

    def test_rtl_trailing_punctuation_joins_rtl_word(self):
        # "שלום," is one segment. First strong char is ש → RTL. The 5
        # graphemes chunk into bigrams ["של","ום",","], reversed to
        # [",","ום","של"]. The comma thus renders leftmost, which is the
        # visual tail of the RTL word.
        units, starts = _segments_to_units("שלום, world")
        assert units == [",", "ום", "של", "wo", "rl", "d"]
        assert starts == [1, 0, 0, 1, 0, 0]


class TestProcessorOutput:
    """End-to-end BigramsRenderingProcessor.__call__ behaviour."""

    def test_strategy_unsupported_raises(self):
        with pytest.raises(NotImplementedError):
            BigramsRenderingProcessor(strategy="words")

    def test_empty_text(self, processor):
        out = processor("", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert out["pixel_values"].shape == (BLOCK_SIZE, BLOCK_SIZE, 3)
        assert out["segment_start_mask"].shape == (1,)

    def test_returns_numpy_arrays(self, processor):
        out = processor("hello", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert isinstance(out["pixel_values"], np.ndarray)
        assert isinstance(out["segment_start_mask"], np.ndarray)
        assert out["segment_start_mask"].dtype == np.int64

    def test_pixel_values_shape(self, processor):
        out = processor("hello world", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        h, w, c = out["pixel_values"].shape
        assert h == BLOCK_SIZE
        assert c == 3
        assert w % BLOCK_SIZE == 0

    def test_mask_length_matches_patch_count(self, processor):
        out = processor("hello world today", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert len(out["segment_start_mask"]) == _patches(out)

    def test_mask_first_entry_is_one(self, processor):
        for text in ["hello", "hello world", "Hello, world.", "สวัสดี hello"]:
            out = processor(text, block_size=BLOCK_SIZE, font_size=FONT_SIZE)
            assert out["segment_start_mask"][0] == 1, text

    def test_mask_counts_segments(self, processor):
        out = processor("hello world", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert out["segment_start_mask"].sum() == 2

    def test_mask_counts_segments_with_punctuation(self, processor):
        # segments: "Hello,", "world."
        out = processor("Hello, world.", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert out["segment_start_mask"].sum() == 2

    def test_whitespace_variants_produce_identical_output(self, processor):
        outputs = [
            processor(t, block_size=BLOCK_SIZE, font_size=FONT_SIZE)
            for t in ["a b", "a  b", "a\tb", "a\nb", "a\u00a0b"]
        ]
        for other in outputs[1:]:
            np.testing.assert_array_equal(outputs[0]["pixel_values"], other["pixel_values"])
            np.testing.assert_array_equal(outputs[0]["segment_start_mask"], other["segment_start_mask"])

    def test_leading_and_trailing_whitespace_ignored(self, processor):
        a = processor("hello world", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        b = processor("  hello\nworld  ", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        np.testing.assert_array_equal(a["pixel_values"], b["pixel_values"])
        np.testing.assert_array_equal(a["segment_start_mask"], b["segment_start_mask"])

    def test_reproducibility_same_input_same_output(self, processor):
        out1 = processor("Hello, world!", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        out2 = processor("Hello, world!", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        np.testing.assert_array_equal(out1["pixel_values"], out2["pixel_values"])
        np.testing.assert_array_equal(out1["segment_start_mask"], out2["segment_start_mask"])

    def test_fontconfig_init_called_before_rendering(self, monkeypatch):
        """When font is provided, fontconfig init must run before rasterisation,
        otherwise patch widths depend on host fonts."""
        calls: list[str] = []

        from mpixel.renderer import processor as processor_mod

        class DummyOwner:
            def __init__(self, font):
                self.font = font

            def _ensure_fontconfig_initialized(self):
                calls.append("ensure_fontconfig")

        def spy_render(*a, **kw):
            calls.append("render")
            units = a[0]
            image = np.zeros((kw.get("block_size", BLOCK_SIZE), BLOCK_SIZE, 3), dtype=np.uint8)
            return image, [1] * len(units)

        monkeypatch.setattr(processor_mod, "PixelRendererProcessor", DummyOwner)
        monkeypatch.setattr(processor_mod, "_render_units", spy_render)

        p = BigramsRenderingProcessor(strategy="bigrams", font="dummy")
        p("hello", block_size=BLOCK_SIZE, font_size=FONT_SIZE)

        assert "ensure_fontconfig" in calls
        assert calls.index("ensure_fontconfig") < calls.index("render")

    def test_no_fontconfig_init_when_font_none(self, monkeypatch):
        """With font=None, the processor must not touch fontconfig at all."""
        import pixel_renderer

        called: list[bool] = []
        monkeypatch.setattr(
            pixel_renderer.PixelRendererProcessor,
            "_ensure_fontconfig_initialized",
            lambda self: called.append(True),
        )

        p = BigramsRenderingProcessor(strategy="bigrams", font=None)
        p("hello", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert called == []


class TestSegmentsToUnitsAdaptive:
    """_segments_to_units with non-default strategy/n."""

    def test_words_strategy_keeps_segment_whole(self):
        units, starts = _segments_to_units("hello world", strategy="words")
        assert units == ["hello", "world"]
        assert starts == [1, 1]

    def test_words_strategy_keeps_punctuation_attached(self):
        units, starts = _segments_to_units("Hello, world.", strategy="words")
        assert units == ["Hello,", "world."]
        assert starts == [1, 1]

    def test_words_strategy_thai_single_unit(self):
        units, starts = _segments_to_units("สวัสดีครับ", strategy="words")
        assert units == ["สวัสดีครับ"]
        assert starts == [1]

    def test_ngrams_n3(self):
        # "helloworld" with n=3 → ["hel", "low", "orl", "d"]
        units, starts = _segments_to_units("helloworld", strategy="ngrams", n=3)
        assert units == ["hel", "low", "orl", "d"]
        assert starts == [1, 0, 0, 0]

    def test_ngrams_n1_per_grapheme(self):
        units, starts = _segments_to_units("hi bye", strategy="ngrams", n=1)
        assert units == ["h", "i", "b", "y", "e"]
        assert starts == [1, 0, 1, 0, 0]

    def test_bigrams_strategy_matches_default(self):
        assert _segments_to_units("hello world", strategy="bigrams") \
            == _segments_to_units("hello world")

    def test_ngrams_n2_matches_bigrams(self):
        assert _segments_to_units("hello world", strategy="ngrams", n=2) \
            == _segments_to_units("hello world", strategy="bigrams")

    def test_words_rtl_reverses_segment_order(self):
        units, starts = _segments_to_units("אב גד", strategy="words")
        assert units == ["גד", "אב"]
        assert starts == [1, 1]

    def test_ngrams_rtl_reverses_chunks(self):
        # "אבגד" with n=2 → logical ["אב", "גד"] → reversed ["גד", "אב"].
        units, starts = _segments_to_units("אבגד", strategy="ngrams", n=2)
        assert units == ["גד", "אב"]
        assert starts == [1, 0]


class TestAdaptiveRenderingProcessor:
    """End-to-end AdaptiveRenderingProcessor behaviour."""

    def test_renders_latin(self):
        p = AdaptiveRenderingProcessor()
        out = p("hello world", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        h, w, c = out["pixel_values"].shape
        assert h == BLOCK_SIZE
        assert c == 3
        assert w % BLOCK_SIZE == 0
        assert len(out["segment_start_mask"]) == _patches(out)
        assert out["segment_start_mask"].sum() == 2

    def test_empty_text(self):
        p = AdaptiveRenderingProcessor()
        out = p("", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert out["pixel_values"].shape == (BLOCK_SIZE, BLOCK_SIZE, 3)
        assert out["segment_start_mask"].shape == (1,)

    def test_returns_numpy_arrays(self):
        p = AdaptiveRenderingProcessor()
        out = p("hello", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        assert isinstance(out["pixel_values"], np.ndarray)
        assert isinstance(out["segment_start_mask"], np.ndarray)
        assert out["segment_start_mask"].dtype == np.int64

    def test_reproducibility(self):
        p = AdaptiveRenderingProcessor()
        a = p("Hello, world!", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        b = p("Hello, world!", block_size=BLOCK_SIZE, font_size=FONT_SIZE)
        np.testing.assert_array_equal(a["pixel_values"], b["pixel_values"])
        np.testing.assert_array_equal(a["segment_start_mask"], b["segment_start_mask"])


class TestAutoStrategyDispatch:
    """_auto_strategy_for_segment picks the right (strategy, n) per segment."""

    def test_latin_segment_falls_back_to_bigrams(self):
        assert _auto_strategy_for_segment("hello") == ("ngrams", 2)

    def test_latin_with_punctuation_bigrams(self):
        assert _auto_strategy_for_segment("Hello,") == ("ngrams", 2)

    def test_devanagari_segment_bigrams(self):
        # Brahmic scripts have only local (intra-cluster) shape dependence;
        # \X already clusters conjuncts → BIGRAMS is the safe default.
        assert _auto_strategy_for_segment("अक्षर") == ("ngrams", 2)

    def test_cyrillic_segment_bigrams(self):
        assert _auto_strategy_for_segment("привет") == ("ngrams", 2)

    def test_thai_segment_bigrams(self):
        assert _auto_strategy_for_segment("สวัสดี") == ("ngrams", 2)

    def test_hebrew_segment_words(self):
        assert _auto_strategy_for_segment("שלום") == ("words", 1)

    def test_arabic_segment_words(self):
        assert _auto_strategy_for_segment("مرحبا") == ("words", 1)

    def test_nko_segment_words(self):
        # N'Ko is connected cursive like Arabic.
        assert _auto_strategy_for_segment("ߒߞߏ") == ("words", 1)

    def test_han_segment_ngrams_n1(self):
        assert _auto_strategy_for_segment("你好") == ("ngrams", 1)

    def test_hiragana_segment_ngrams_n1(self):
        assert _auto_strategy_for_segment("こんにちは") == ("ngrams", 1)

    def test_katakana_segment_ngrams_n1(self):
        assert _auto_strategy_for_segment("カタカナ") == ("ngrams", 1)

    def test_hangul_segment_ngrams_n1(self):
        assert _auto_strategy_for_segment("한국어") == ("ngrams", 1)

    def test_mixed_kanji_kana_segment_ngrams_n1(self):
        # Japanese sentence without whitespace — any em-width char triggers n=1.
        assert _auto_strategy_for_segment("日本語はむずかしい") == ("ngrams", 1)

    def test_latin_mixed_into_japanese_segment_ngrams_n1(self):
        # "iPhoneを使う" — single segment (no whitespace), mixed Latin+kana+kanji.
        # Em-width char present → n=1 for the whole segment.
        assert _auto_strategy_for_segment("iPhoneを使う") == ("ngrams", 1)

    def test_cursive_wins_over_em_width(self):
        # Pathological mixed segment — cursive shape correctness trumps.
        assert _auto_strategy_for_segment("مرحبا你好") == ("words", 1)


class TestAutoStrategyEndToEnd:
    """_segments_to_units and the processor under strategy='auto'."""

    def test_segments_to_units_auto_latin(self):
        # "hello world" — two Latin segments, both bigrammed.
        units, starts = _segments_to_units("hello world", strategy="auto")
        assert units == ["he", "ll", "o", "wo", "rl", "d"]
        assert starts == [1, 0, 0, 1, 0, 0]

    def test_segments_to_units_auto_devanagari(self):
        # अक्षर → 3 clusters [अ, क्ष, र] → bigrams [अक्ष, र]. Conjunct preserved.
        units, starts = _segments_to_units("अक्षर", strategy="auto")
        assert units == ["अक्ष", "र"]
        assert starts == [1, 0]

    def test_segments_to_units_auto_arabic_is_words(self):
        # Arabic segment becomes one unit under auto (cursive → words).
        units, starts = _segments_to_units("مرحبا", strategy="auto")
        assert units == ["مرحبا"]
        assert starts == [1]

    def test_segments_to_units_auto_han_is_n1(self):
        units, starts = _segments_to_units("你好世界", strategy="auto")
        assert units == ["你", "好", "世", "界"]
        assert starts == [1, 0, 0, 0]

    def test_segments_to_units_auto_hangul_is_n1(self):
        units, starts = _segments_to_units("한국어", strategy="auto")
        assert units == ["한", "국", "어"]
        assert starts == [1, 0, 0]

    def test_segments_to_units_auto_mixed_script_per_segment(self):
        # Four whitespace-delimited segments, four different dispatch paths.
        # LTR ordering: "hello" (bigrams) → "你好" (n=1) → "한국" (n=1) → "अक्षर" (bigrams).
        # RTL-only Arabic segment would reorder; tested separately.
        units, starts = _segments_to_units(
            "hello 你好 한국 अक्षर", strategy="auto"
        )
        assert units == [
            "he", "ll", "o",          # hello: bigrams
            "你", "好",                # 你好: n=1
            "한", "국",                # 한국: n=1
            "अक्ष", "र",                # अक्षर: bigrams (conjunct kept)
        ]
        assert starts == [1, 0, 0, 1, 0, 1, 0, 1, 0]

    def test_segments_to_units_auto_arabic_rtl_reorder(self):
        # Two Arabic segments reorder in RTL run; each becomes one unit (words).
        units, starts = _segments_to_units(
            "مرحبا العالم", strategy="auto"
        )
        assert units == ["العالم", "مرحبا"]
        assert starts == [1, 1]

    def test_processor_auto_latin_matches_bigrams(self):
        text = "hello world today"
        a = AdaptiveRenderingProcessor()(
            text, block_size=BLOCK_SIZE, font_size=FONT_SIZE
        )
        b = BigramsRenderingProcessor()(
            text, block_size=BLOCK_SIZE, font_size=FONT_SIZE
        )
        np.testing.assert_array_equal(a["pixel_values"], b["pixel_values"])
        np.testing.assert_array_equal(
            a["segment_start_mask"], b["segment_start_mask"]
        )

    def test_processor_auto_arabic_renders(self):
        out = AdaptiveRenderingProcessor()(
            "مرحبا العالم", block_size=BLOCK_SIZE, font_size=FONT_SIZE
        )
        assert out["segment_start_mask"].sum() == 2

    def test_processor_auto_han_renders(self):
        out = AdaptiveRenderingProcessor()(
            "你好世界", block_size=BLOCK_SIZE, font_size=FONT_SIZE
        )
        assert len(out["segment_start_mask"]) == _patches(out)

    def test_processor_auto_mixed_script_renders(self):
        out = AdaptiveRenderingProcessor()(
            "hello 你好 한국 अक्षर", block_size=BLOCK_SIZE, font_size=FONT_SIZE
        )
        assert out["segment_start_mask"].sum() == 4
        assert len(out["segment_start_mask"]) == _patches(out)
