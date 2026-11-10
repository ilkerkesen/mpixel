"""Tests for scripts/data/download_fineweb2.py — adaptive rendering download orchestration."""

import importlib.util
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def download_module():
    """Import the download script as a module (it's not a package)."""
    script_path = str(Path(__file__).parents[2] / "scripts" / "data" / "download_fineweb2.py")
    spec = importlib.util.spec_from_file_location("download_fineweb2", script_path)
    assert spec is not None and spec.loader is not None, f"Could not find module at {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestParseArgs:
    """CLI argument parsing for the download script."""

    def test_defaults(self, download_module):
        """Only the positional language arg is required; everything else has defaults."""
        args = download_module.parse_args(["eng_Latn"])
        assert args.language == "eng_Latn"
        assert args.num_train_patches == 0
        assert args.compression == "zstd"
        assert args.num_patches == 512

    def test_custom_values(self, download_module):
        """All flags can be overridden from the command line."""
        args = download_module.parse_args([
            "fin_Latn",
            "--num-train-patches", "100000",
            "--compression", "none",
            "--num-patches", "1024",
            "--seed", "42",
        ])
        assert args.language == "fin_Latn"
        assert args.num_train_patches == 100000
        assert args.compression == "none"
        assert args.num_patches == 1024
        assert args.seed == 42

    def test_missing_language_exits(self, download_module):
        """Language is a required positional argument."""
        with pytest.raises(SystemExit):
            download_module.parse_args([])

    def test_invalid_compression_exits(self, download_module):
        """Only 'none' and 'gzip' are valid compression choices."""
        with pytest.raises(SystemExit):
            download_module.parse_args(["eng_Latn", "--compression", "bzip2"])


class TestStreamFineweb2:
    """Integration of stream_fineweb2 with mocked external dependencies
    (load_dataset, unit patch measurement, and _stream_to_shards)."""

    def _patch_deps(self, mod):
        """Return three context-manager patches for the heavy dependencies."""
        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = MagicMock()

        return (
            patch.object(mod, "load_dataset", return_value=mock_dataset),
            patch.object(mod, "create_measure_unit_patches_fn", return_value=lambda u: 2),
            patch.object(mod, "_stream_to_shards", return_value=(1000, 500, 3000, 10)),
        )

    def test_train_only_calls_stream_with_correct_args(self, download_module, tmp_path):
        """Train-only run: _stream_to_shards called once with train split,
        and load_dataset uses English fineweb config with streaming=True."""
        p1, p2, p3 = self._patch_deps(download_module)
        with p1 as mock_ld, p2, p3 as mock_stream:
            download_module.stream_fineweb2(
                language="eng_Latn", num_train_patches=1000, output_dir=tmp_path,
            )
            assert mock_stream.call_count == 1
            kwargs = mock_stream.call_args.kwargs
            assert kwargs["split_name"] == "train"
            assert kwargs["max_patches"] == 1000
            assert kwargs["file_prefix"] == "fineweb-eng_Latn-train"
            assert kwargs["compression"] == "gzip"
            assert kwargs["num_patches"] == 1024
            mock_ld.assert_called_once()
            ld_args, ld_kwargs = mock_ld.call_args
            assert ld_args[0] == "HuggingFaceFW/fineweb"
            assert ld_args[1] == "sample-100BT"
            assert ld_kwargs.get("streaming") is True

    def test_non_english_uses_fineweb2(self, download_module, tmp_path):
        """Non-English languages use HuggingFaceFW/fineweb-2 with the language
        code as dataset config, and streaming=True."""
        p1, p2, p3 = self._patch_deps(download_module)
        with p1 as mock_ld, p2, p3 as mock_stream:
            download_module.stream_fineweb2(
                language="fin_Latn", num_train_patches=1000, output_dir=tmp_path,
            )
            ld_args, ld_kwargs = mock_ld.call_args
            assert ld_args[0] == "HuggingFaceFW/fineweb-2"
            assert ld_args[1] == "fin_Latn"
            assert ld_kwargs.get("streaming") is True
            assert mock_stream.call_args.kwargs["file_prefix"] == "fineweb2-fin_Latn-train"

    def test_patches_per_file_zero_raises(self, download_module, tmp_path):
        """patches_per_file must be positive — no mocks needed, fails immediately."""
        with pytest.raises(ValueError, match="patches_per_file must be positive"):
            download_module.stream_fineweb2(
                language="eng_Latn", num_train_patches=1000, patches_per_file=0, output_dir=tmp_path,
            )

    def test_measure_fn_passed_to_stream(self, download_module, tmp_path):
        """The measure function from create_measure_unit_patches_fn is forwarded
        to _stream_to_shards."""
        sentinel_fn = lambda u: 3  # noqa: E731
        p1, _, p3 = self._patch_deps(download_module)
        p2 = patch.object(download_module, "create_measure_unit_patches_fn", return_value=sentinel_fn)
        with p1, p2, p3 as mock_stream:
            download_module.stream_fineweb2(
                language="eng_Latn", num_train_patches=1000, output_dir=tmp_path,
            )
            assert mock_stream.call_args.kwargs["measure_fn"] is sentinel_fn

    def test_stream_returns_four_tuple(self, download_module, tmp_path):
        """_stream_to_shards returns (total_patches, total_words, total_chars, num_examples)."""
        p1, p2, p3 = self._patch_deps(download_module)
        with p1, p2, p3 as mock_stream:
            mock_stream.return_value = (5000, 2500, 15000, 50)
            download_module.stream_fineweb2(
                language="eng_Latn", num_train_patches=5000, output_dir=tmp_path,
            )
            patches, words, chars, docs = mock_stream.return_value
            assert patches == 5000
            assert words == 2500
            assert chars == 15000
            assert docs == 50


class TestSplitTextByPatches:
    """Unit tests for adaptive text splitting by patch count."""

    def test_single_chunk_when_fits(self, download_module):
        """All units fit within max_patches → single chunk."""
        mock_segments = MagicMock(return_value=(["He", "ll", "o"], [1, 0, 0]))
        measure = lambda u: 1  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("Hello", measure, max_patches=10)
        assert len(chunks) == 1
        assert chunks[0]['num_patches'] == 3
        assert chunks[0]['num_words'] == 1
        assert chunks[0]['num_chars'] == 5
        assert chunks[0]['text'] == "Hello"

    def test_multi_chunk_split(self, download_module):
        """Units exceeding max_patches are split across chunks."""
        units = ["ab", "cd", "ef", "gh"]
        starts = [1, 0, 1, 0]
        mock_segments = MagicMock(return_value=(units, starts))
        measure = lambda u: 3  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("abcd efgh", measure, max_patches=6)
        assert len(chunks) == 2
        assert chunks[0]['num_patches'] == 6
        assert chunks[0]['num_words'] == 1
        assert chunks[0]['text'] == "abcd"
        assert chunks[1]['num_patches'] == 6
        assert chunks[1]['num_words'] == 1
        assert chunks[1]['text'] == "efgh"

    def test_counts_words_from_segment_starts(self, download_module):
        """Word count is derived from segment_start flags, not whitespace."""
        units = ["He", "ll", "o", "wo", "rl", "d"]
        starts = [1, 0, 0, 1, 0, 0]
        mock_segments = MagicMock(return_value=(units, starts))
        measure = lambda u: 1  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("Hello world", measure, max_patches=100)
        assert len(chunks) == 1
        assert chunks[0]['num_words'] == 2
        assert chunks[0]['num_chars'] == 10
        assert chunks[0]['text'] == "Hello world"

    def test_empty_text(self, download_module):
        """Empty text returns no chunks."""
        mock_segments = MagicMock(return_value=([], []))
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("", lambda u: 1, max_patches=10)
        assert chunks == []

    def test_oversized_unit(self, download_module):
        """A single unit exceeding max_patches is not split further."""
        units = ["longword"]
        starts = [1]
        mock_segments = MagicMock(return_value=(units, starts))
        measure = lambda u: 20  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("longword", measure, max_patches=5)
        assert len(chunks) == 1
        assert chunks[0]['num_patches'] == 20
        assert chunks[0]['num_words'] == 1

    def test_text_reconstruction_preserves_segments(self, download_module):
        """Text is reconstructed with spaces between segments, not between units
        within the same segment."""
        units = ["He", "ll", "o", "wo", "rl", "d", "!"]
        starts = [1, 0, 0, 1, 0, 0, 1]
        mock_segments = MagicMock(return_value=(units, starts))
        measure = lambda u: 1  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("Hello world !", measure, max_patches=100)
        assert chunks[0]['text'] == "Hello world !"
        assert chunks[0]['num_words'] == 3

    def test_chunk_boundary_respects_segment(self, download_module):
        """When a chunk splits mid-text, the second chunk starts a new context."""
        units = ["ab", "cd", "ef", "gh", "ij"]
        starts = [1, 0, 1, 0, 1]
        mock_segments = MagicMock(return_value=(units, starts))
        measure = lambda u: 2  # noqa: E731
        with patch("mpixel.data.fineweb._text_to_units", mock_segments):
            chunks = download_module.split_text_by_patches("abcd efgh ij", measure, max_patches=4)
        assert len(chunks) == 3
        assert chunks[0]['text'] == "abcd"
        assert chunks[0]['num_patches'] == 4
        assert chunks[1]['text'] == "efgh"
        assert chunks[1]['num_patches'] == 4
        assert chunks[2]['text'] == "ij"
        assert chunks[2]['num_patches'] == 2
