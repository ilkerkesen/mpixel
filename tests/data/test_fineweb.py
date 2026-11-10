"""Tests for mpixel.data.fineweb — pure data utilities for FineWeb sharding."""

import json
import gzip
import pytest

from mpixel.data.fineweb import (
    get_unit,
    count_units,
    split_text_by_width_binary,
    split_text_by_width_fast,
    open_output_file,
    _stream_to_shards,
)


class TestGetUnit:
    """Map language codes to their counting unit (words, chars, or 4-grams)."""

    @pytest.mark.parametrize("lang,expected", [
        ("eng_Latn", "words"),
        ("fin_Latn", "words"),
        ("ukr_Cyrl", "words"),
        ("arb_Arab", "words"),
        ("cmn_Hani", "chars"),
        ("jpn_Jpan", "chars"),
        ("kor_Hang", "chars"),
        ("tha_Thai", "4-grams"),
    ])
    def test_language_to_unit(self, lang, expected):
        assert get_unit(lang) == expected

    def test_unknown_language_raises(self):
        """Unsupported language codes must be rejected."""
        with pytest.raises(Exception):
            get_unit("xyz_Unknown")


class TestCountUnits:
    """Count words, characters, or 4-grams in a text string."""

    def test_words(self):
        assert count_units("hello world foo", "words") == 3

    def test_words_empty(self):
        assert count_units("", "words") == 0

    def test_chars(self):
        assert count_units("abcdef", "chars") == 6

    def test_chars_unicode(self):
        """CJK characters are each counted as one unit."""
        assert count_units("哥本哈根", "chars") == 4

    def test_4grams(self):
        assert count_units("abcdefgh", "4-grams") == 2

    def test_4grams_remainder(self):
        """Partial trailing 4-gram is dropped (integer division)."""
        assert count_units("abcde", "4-grams") == 1

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit type"):
            count_units("hello", "syllables")


class TestSplitTextByWidthBinary:
    """Binary-search text splitting: every chunk's pixel width <= max_width."""

    @pytest.fixture
    def measure(self):
        return lambda text: len(text) * 8

    def test_empty_returns_empty(self, measure):
        assert split_text_by_width_binary("", measure, 100, "words") == []

    def test_fits_single_chunk(self, measure):
        """Text that already fits should come back as a single chunk."""
        result = split_text_by_width_binary("hello world", measure, 200, "words")
        assert result == ["hello world"]

    def test_splits_and_preserves_content(self, measure):
        """Chunks must rejoin to the original text and each fit within max_width."""
        text = "one two three four five"
        result = split_text_by_width_binary(text, measure, 40, "words")
        assert len(result) > 1
        assert " ".join(result) == text
        for chunk in result:
            assert measure(chunk) <= 40

    def test_chars_unit(self, measure):
        """With chars unit (8px each), max_width=24 should yield 3-char chunks."""
        result = split_text_by_width_binary("abcdefgh", measure, 24, "chars")
        assert result == ["abc", "def", "gh"]

    def test_single_oversized_token_still_emitted(self, measure):
        """A single token wider than max_width is emitted anyway (best=1 fallback)."""
        result = split_text_by_width_binary("superlongword", measure, 10, "words")
        assert result == ["superlongword"]


class TestSplitTextByWidthFast:
    """Proportional-approximation splitting (faster, may slightly exceed max_width)."""

    @pytest.fixture
    def measure(self):
        return lambda text: len(text) * 8

    def test_empty_returns_empty(self, measure):
        assert split_text_by_width_fast("", measure, 100, "words") == []

    def test_fits_single_chunk(self, measure):
        result = split_text_by_width_fast("hello world", measure, 200, "words")
        assert result == ["hello world"]

    def test_all_tokens_preserved(self, measure):
        """All words from the input must appear in the output chunks."""
        text = "one two three four five"
        result = split_text_by_width_fast(text, measure, 40, "words")
        assert len(result) > 1
        all_words = []
        for chunk in result:
            all_words.extend(chunk.split())
        assert all_words == text.split()


class TestOpenOutputFile:
    """Shard file creation with correct naming and compression."""

    def test_plain_jsonl(self, tmp_path):
        """Uncompressed shard: .jsonl extension, writable."""
        f, path = open_output_file(tmp_path, "shard", 0, "none")
        f.write("test\n")
        f.close()
        assert path.name == "shard-00000000.jsonl"
        assert path.read_text() == "test\n"

    def test_gzip_jsonl(self, tmp_path):
        """Gzip shard: .jsonl.gz extension, content readable after decompression."""
        f, path = open_output_file(tmp_path, "shard", 0, "gzip")
        f.write("test\n")
        f.close()
        assert path.name == "shard-00000000.jsonl.gz"
        with gzip.open(path, "rt") as g:
            assert g.read() == "test\n"

    def test_creates_parent_dirs(self, tmp_path):
        """Missing parent directories are created automatically."""
        nested = tmp_path / "a" / "b"
        f, path = open_output_file(nested, "shard", 0, "none")
        f.close()
        assert nested.exists()


class TestStreamToShards:
    """Stream dataset examples into sharded JSONL files with metadata.

    The measure fixture gives one patch per unit, and each one-letter word is
    one unit, so patches == words in these tests.
    """

    def _make_iter(self, texts):
        return iter([{"text": t, "id": str(i)} for i, t in enumerate(texts)])

    def _common_kwargs(self, measure):
        return dict(
            language="eng_Latn",
            compression="none",
            measure_fn=measure,
            num_patches=512,
            text_column="text",
        )

    def test_basic_streaming(self, tmp_path, mock_measure_fn):
        """Two docs with 2+3 words produce 2 examples and 5 total patches."""
        output_dir = tmp_path / "out"
        total_patches, total_words, _, n = _stream_to_shards(
            self._make_iter(["a b", "c d e"]),
            max_patches=0,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=1_000_000,
            split_name="train",
            metadata={},
            **self._common_kwargs(mock_measure_fn),
        )
        assert n == 2
        assert total_patches == 5
        assert total_words == 5

    def test_metadata_written(self, tmp_path, mock_measure_fn):
        """A metadata JSON sidecar is written with correct counts."""
        output_dir = tmp_path / "out"
        _stream_to_shards(
            self._make_iter(["a b"]),
            max_patches=0,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=1_000_000,
            split_name="train",
            metadata={"format": "test-v1"},
            **self._common_kwargs(mock_measure_fn),
        )
        meta = json.loads((output_dir / "test-metadata.json").read_text())
        assert meta["num_examples"] == 1
        assert meta["num_shards"] == 1
        assert meta["total_patches"] == 2
        assert meta["format"] == "test-v1"

    def test_max_patches_stops_early(self, tmp_path, mock_measure_fn):
        """With 1-patch docs, streaming stops exactly at max_patches."""
        output_dir = tmp_path / "out"
        total_patches, _, _, n = _stream_to_shards(
            self._make_iter(["x"] * 100),
            max_patches=10,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=1_000_000,
            split_name="train",
            metadata={},
            **self._common_kwargs(mock_measure_fn),
        )
        assert total_patches == 10
        assert n == 10

    def test_shard_rotation(self, tmp_path, mock_measure_fn):
        """When patches_per_file is exceeded, a new shard opens. No shard is empty
        and all input texts appear across shards."""
        output_dir = tmp_path / "out"
        texts = ["a b", "c d", "e f"]
        _stream_to_shards(
            self._make_iter(texts),
            max_patches=0,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=3,
            split_name="train",
            metadata={},
            **self._common_kwargs(mock_measure_fn),
        )
        jsonl_files = sorted(output_dir.glob("test-*.jsonl"))
        assert len(jsonl_files) >= 2
        all_texts = []
        for shard in jsonl_files:
            lines = [l for l in shard.read_text().splitlines() if l.strip()]
            assert len(lines) > 0, f"Empty shard: {shard.name}"
            for line in lines:
                all_texts.append(json.loads(line)["text"])
        for t in texts:
            assert t in all_texts

    def test_max_patches_overshoot(self, tmp_path, mock_measure_fn):
        """A multi-patch doc can push total_patches past max_patches (checked after
        counting), so we expect overshoot: 2 docs x 5 patches = 10 > max_patches=7."""
        output_dir = tmp_path / "out"
        total_patches, _, _, n = _stream_to_shards(
            self._make_iter(["a b c d e", "f g h i j", "k l m n o"]),
            max_patches=7,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=1_000_000,
            split_name="train",
            metadata={},
            **self._common_kwargs(mock_measure_fn),
        )
        assert n == 2
        assert total_patches == 10

    def test_skips_empty_text(self, tmp_path, mock_measure_fn):
        """Empty strings and None values are silently skipped."""
        output_dir = tmp_path / "out"
        iterator = iter([{"text": "", "id": "0"}, {"text": "hello", "id": "1"}, {"text": None, "id": "2"}])
        _, _, _, n = _stream_to_shards(
            iterator,
            max_patches=0,
            output_dir=output_dir,
            file_prefix="test",
            patches_per_file=1_000_000,
            split_name="train",
            metadata={},
            **self._common_kwargs(mock_measure_fn),
        )
        assert n == 1
