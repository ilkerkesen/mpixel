"""Tests for mpixel.processing — turn text into MPIXEL model inputs.

The reference is the pretraining pipeline (`preprocess_text_for_pretraining`), with the same
renderer, fonts and transforms as training. The processor must give exactly the same inputs.
examples/run_sib200.py also renders with the processor, so it must give the same inputs as the
SIB-200 data module (`preprocess_text_for_classification`) too.
"""
import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from font_download import FontConfig

from mpixel.datamodules.pretraining import FineWebIterableDataModule, preprocess_text_for_pretraining
from mpixel.datamodules.sib200 import SIB200DataModule, preprocess_text_for_classification
from mpixel.models.masking import SpanMaskingGenerator
from mpixel.processing import CONFIG_NAME, MPIXELProcessor
from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED
from mpixel.renderer.processor import AdaptiveRenderingProcessor
from mpixel.utils.preprocessing import get_transforms


# Settings of the released model (mpixel); the same as the pretraining defaults.
PATCH_SIZE = 16
MAX_NUM_PATCHES = 512
FONT_SIZE = 12

TEXTS = [
    "Hello world!",
    "Merhaba dünya, nasılsın?",
    "مرحبا بالعالم",
    "你好，世界。",
    "नमस्ते दुनिया",
    "Ελληνικά και English mixed 123.",
    "བཀྲ་ཤིས་བདེ་ལེགས།",
    # Longer than 512 patches. Much longer texts fail in cairo (image too wide), also in pretraining.
    "This text is longer than the maximum number of patches. " * 30,
]


def reference_inputs(text: str) -> dict:
    """Inputs exactly as the pretraining data module builds them (see FineWebIterableDataset)."""
    out = preprocess_text_for_pretraining(
        text=text,
        pixel_processor=AdaptiveRenderingProcessor(font=FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)),
        transforms=get_transforms(do_resize=False),
        patch_size=PATCH_SIZE,
        max_num_patches=MAX_NUM_PATCHES,
        mask_generator=SpanMaskingGenerator(
            max_span_length=6, spacing="span", cumulative_span_weights=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        ),
        span_mask_ratio=0.25,
        font_size=FONT_SIZE,
    )
    return {"pixel_values": out["pixel_values"], "attention_mask": out["attention_mask"]}


def assert_same_pixels(actual: torch.Tensor, expected: torch.Tensor):
    """Pixel-by-pixel comparison with no tolerance."""
    assert actual.shape == expected.shape
    different = (actual != expected).nonzero()
    assert len(different) == 0, (
        f"{len(different)} of {expected.numel()} values are different; "
        f"first at (channel, row, column) {tuple(different[0].tolist())}"
    )


@pytest.fixture(scope="module")
def processor():
    return MPIXELProcessor(patch_size=PATCH_SIZE, max_num_patches=MAX_NUM_PATCHES, font_size=FONT_SIZE)


class TestSameAsPretraining:
    @pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
    def test_single_text(self, processor, text):
        inputs = processor(text)
        expected = reference_inputs(text)

        assert_same_pixels(inputs["pixel_values"][0], expected["pixel_values"])
        assert torch.equal(inputs["attention_mask"][0], expected["attention_mask"])

    def test_batch(self, processor):
        inputs = processor(TEXTS)

        for i, text in enumerate(TEXTS):
            expected = reference_inputs(text)
            assert_same_pixels(inputs["pixel_values"][i], expected["pixel_values"])
            assert torch.equal(inputs["attention_mask"][i], expected["attention_mask"]), text


class TestSameAsSIB200:
    """With the SIB200DataModule defaults, the processor gives the inputs of the SIB-200 data module."""

    @pytest.fixture(scope="class")
    def settings(self):
        defaults = inspect.signature(SIB200DataModule.__init__).parameters
        assert defaults["renderer_type"].default == "adaptive"
        return {key: defaults[key].default for key in ("patch_size", "max_num_patches", "font_size")}

    @pytest.fixture(scope="class")
    def sib200_processor(self, settings):
        return MPIXELProcessor(**settings)

    @pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
    def test_single_text(self, settings, sib200_processor, text):
        inputs = sib200_processor(text)
        expected = preprocess_text_for_classification(
            text=text,
            pixel_processor=AdaptiveRenderingProcessor(font=FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)),
            transforms=get_transforms(do_resize=False),
            **settings,
        )

        assert_same_pixels(inputs["pixel_values"][0], expected["pixel_values"])
        assert torch.equal(inputs["attention_mask"][0], expected["attention_mask"])


class TestMPIXELProcessor:
    def test_output_shapes(self, processor):
        inputs = processor(TEXTS[:2])

        assert set(inputs) == {"pixel_values", "attention_mask"}
        assert inputs["pixel_values"].shape == (2, 3, PATCH_SIZE, MAX_NUM_PATCHES * PATCH_SIZE)
        assert inputs["attention_mask"].shape == (2, MAX_NUM_PATCHES)

    def test_single_string_gives_batch_of_one(self, processor):
        inputs = processor("Hello world!")
        assert inputs["pixel_values"].shape == (1, 3, PATCH_SIZE, MAX_NUM_PATCHES * PATCH_SIZE)

    def test_renderer_is_built_once(self):
        renderer_cls = MagicMock(wraps=AdaptiveRenderingProcessor)
        with patch("mpixel.processing.AdaptiveRenderingProcessor", renderer_cls):
            processor = MPIXELProcessor()
            processor(TEXTS[:2])
            processor(TEXTS[0])
        assert renderer_cls.call_count == 1

    def test_uses_pretraining_fonts(self):
        """Fonts are registered for the whole process, so the pixel tests cannot find a wrong font set."""
        with patch("mpixel.processing.FontConfig") as font_config:
            MPIXELProcessor()
        font_config.assert_called_once_with(sources=FONTS_NOTO_SANS_EXTENDED)

    def test_defaults_match_pretraining_defaults(self):
        """Without a config file, the processor must use the pretraining defaults."""
        datamodule_defaults = inspect.signature(FineWebIterableDataModule.__init__).parameters
        assert datamodule_defaults["renderer_name"].default == "adaptive"
        for key, value in MPIXELProcessor().to_dict().items():
            assert value == datamodule_defaults[key].default, key


class TestSaveAndLoad:
    @pytest.fixture
    def saved_dir(self, tmp_path):
        MPIXELProcessor(patch_size=24, max_num_patches=256, font_size=16).save_pretrained(tmp_path)
        return tmp_path

    def test_save_pretrained(self, saved_dir):
        assert json.loads((saved_dir / CONFIG_NAME).read_text()) == {
            "patch_size": 24,
            "max_num_patches": 256,
            "font_size": 16,
        }

    def test_from_pretrained_local_directory(self, saved_dir):
        loaded = MPIXELProcessor.from_pretrained(saved_dir)
        assert loaded.to_dict() == json.loads((saved_dir / CONFIG_NAME).read_text())

    def test_from_pretrained_hub_id(self, saved_dir):
        with patch("mpixel.processing.hf_hub_download", return_value=str(saved_dir / CONFIG_NAME)) as download:
            loaded = MPIXELProcessor.from_pretrained("org/mpixel")

        download.assert_called_once_with(repo_id="org/mpixel", filename=CONFIG_NAME)
        assert loaded.to_dict() == json.loads((saved_dir / CONFIG_NAME).read_text())
