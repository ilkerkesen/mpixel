"""Tests for mpixel.datamodules.pretraining — Lightning DataModule for MAE pretraining."""

import math
import pytest
import torch
from unittest.mock import patch, MagicMock

from mpixel.datamodules.pretraining import (
    preprocess_text_for_pretraining,
    FineWebDatasetV1,
    FineWebDataModuleV1,
)


class TestPreprocessText:
    """Render text to pixel values and produce attention/patch masks."""

    PATCH_SIZE = 24
    MAX_NUM_PATCHES = 64  # small for fast tests

    def _call(self, text, processor, transforms, mask_gen):
        return preprocess_text_for_pretraining(
            text=text,
            pixel_processor=processor,
            transforms=transforms,
            patch_size=self.PATCH_SIZE,
            max_num_patches=self.MAX_NUM_PATCHES,
            mask_generator=mask_gen,
            span_mask_ratio=0.25,
            font_size=16,
        )

    def test_output_keys(self, mock_pixel_processor, mock_transforms, mock_mask_generator):
        """Returned dict must contain exactly these four keys."""
        result = self._call("hello world", mock_pixel_processor, mock_transforms, mock_mask_generator)
        assert set(result.keys()) == {"pixel_values", "attention_mask", "num_text_patches", "patch_mask"}

    def test_pixel_values_shape(self, mock_pixel_processor, mock_transforms, mock_mask_generator):
        """pixel_values is (3, patch_size, max_num_patches * patch_size) — padded to max width."""
        result = self._call("hello world", mock_pixel_processor, mock_transforms, mock_mask_generator)
        pv = result["pixel_values"]
        assert pv.shape == (3, self.PATCH_SIZE, self.MAX_NUM_PATCHES * self.PATCH_SIZE)

    def test_attention_mask_shape_and_values(self, mock_pixel_processor, mock_transforms, mock_mask_generator):
        """Attention mask is 1 for text+SEP region and 0 for padding."""
        result = self._call("hello world", mock_pixel_processor, mock_transforms, mock_mask_generator)
        am = result["attention_mask"]
        n = result["num_text_patches"]
        assert am.shape == (self.MAX_NUM_PATCHES,)
        assert am[:n + 1].sum() == n + 1
        assert am[n + 1:].sum() == 0

    def test_patch_mask_shape_and_padding_zero(self, mock_pixel_processor, mock_transforms, mock_mask_generator):
        """Patch mask covers only the text+SEP region; padding is always 0.
        Also verifies the mask generator receives correct arguments."""
        result = self._call("hello world", mock_pixel_processor, mock_transforms, mock_mask_generator)
        pm = result["patch_mask"]
        n = result["num_text_patches"]
        effective = n + 1  # text + SEP
        assert pm.shape == (self.MAX_NUM_PATCHES,)
        assert pm.dtype in (torch.int32, torch.int64, torch.float32)
        assert pm[effective:].sum() == 0
        expected_to_mask = math.ceil(effective * 0.25)
        mock_mask_generator.assert_called_once_with(
            num_patches_total=effective,
            num_text_patches=n + 1,
            num_patches_to_mask=expected_to_mask,
        )

    def test_num_text_patches_positive(self, mock_pixel_processor, mock_transforms, mock_mask_generator):
        """Non-empty text must produce at least one text patch."""
        result = self._call("hello world test", mock_pixel_processor, mock_transforms, mock_mask_generator)
        assert result["num_text_patches"] > 0


class TestFineWebDatasetV1:
    """Wraps raw data rows with on-the-fly rendering and masking."""

    @patch("mpixel.datamodules.pretraining.FontConfig")
    @patch("mpixel.datamodules.pretraining.FONTS_NOTO_SANS_EXTENDED", new=[])
    @patch("mpixel.datamodules.pretraining.PixelRendererProcessor")
    @patch("mpixel.datamodules.pretraining.get_transforms")
    def test_len(self, mock_gt, mock_rp, mock_fc):
        """Dataset length matches the underlying data."""
        mock_gt.return_value = MagicMock()
        data = [{"text": "a"}, {"text": "b"}]
        ds = FineWebDatasetV1(data=data, split_name="train", patch_size=24, max_num_patches=64, font_size=16)
        assert len(ds) == 2

    @patch("mpixel.datamodules.pretraining.FontConfig")
    @patch("mpixel.datamodules.pretraining.FONTS_NOTO_SANS_EXTENDED", new=[])
    @patch("mpixel.datamodules.pretraining.PixelRendererProcessor")
    @patch("mpixel.datamodules.pretraining.get_transforms")
    @patch("mpixel.datamodules.pretraining.preprocess_text_for_pretraining")
    def test_getitem_retries_on_error(self, mock_preprocess, mock_gt, mock_rp, mock_fc):
        """When rendering fails, __getitem__ advances to the next index and retries."""
        mock_gt.return_value = MagicMock()
        mock_preprocess.side_effect = [
            RuntimeError("render failed"),
            {"pixel_values": torch.zeros(3, 24, 64 * 24), "attention_mask": torch.zeros(64),
             "num_text_patches": 5, "patch_mask": torch.zeros(64)},
        ]
        data = [{"text": "a"}, {"text": "b"}]
        ds = FineWebDatasetV1(data=data, split_name="train", patch_size=24, max_num_patches=64, font_size=16)
        item = ds[0]
        assert mock_preprocess.call_count == 2
        assert "pixel_values" in item
        # Retry should advance to index (0+1)%2 = 1, so we get item "b"
        assert item["text"] == "b"


class TestFineWebDataModuleV1:
    """Lightning DataModule: shard discovery, caching, collation, and dataloaders."""

    @patch("mpixel.datamodules.pretraining.load_dataset")
    def test_prepare_data_skips_if_cache_exists(self, mock_load, tmp_path):
        """When the cache directory already exists, no dataset loading happens."""
        (tmp_path / "cache").mkdir()
        dm = FineWebDataModuleV1(data_dir=str(tmp_path))
        dm.prepare_data()
        mock_load.assert_not_called()

    @patch("mpixel.datamodules.pretraining.load_dataset")
    def test_prepare_data_skips_if_download_missing(self, mock_load, tmp_path):
        """When the download directory is absent, prepare_data is a no-op."""
        dm = FineWebDataModuleV1(data_dir=str(tmp_path))
        dm.prepare_data()
        mock_load.assert_not_called()

    @patch("mpixel.datamodules.pretraining.load_dataset")
    def test_prepare_data_skips_if_no_shards(self, mock_load, tmp_path):
        """When the download directory has no matching shards, prepare_data is a no-op."""
        (tmp_path / "download").mkdir()
        dm = FineWebDataModuleV1(data_dir=str(tmp_path))
        dm.prepare_data()
        mock_load.assert_not_called()

    def test_collate_fn(self, tmp_path):
        """Collation stacks pixel_values, attention_mask, and patch_mask into batches."""
        dm = FineWebDataModuleV1(data_dir=str(tmp_path))
        examples = [
            {"pixel_values": torch.ones(3, 24, 100), "attention_mask": torch.ones(10), "patch_mask": torch.zeros(10)},
            {"pixel_values": torch.zeros(3, 24, 100), "attention_mask": torch.zeros(10), "patch_mask": torch.ones(10)},
        ]
        batch = dm.collate_fn(examples)
        assert batch["pixel_values"].shape == (2, 3, 24, 100)
        assert batch["attention_mask"].shape == (2, 10)
        assert batch["patch_mask"].shape == (2, 10)

    def test_collate_fn_no_patch_mask(self, tmp_path):
        """When examples lack patch_mask, collation omits it from the batch."""
        dm = FineWebDataModuleV1(data_dir=str(tmp_path))
        examples = [
            {"pixel_values": torch.ones(3, 24, 100), "attention_mask": torch.ones(10)},
            {"pixel_values": torch.zeros(3, 24, 100), "attention_mask": torch.zeros(10)},
        ]
        batch = dm.collate_fn(examples)
        assert "patch_mask" not in batch
