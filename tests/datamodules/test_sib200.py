from unittest.mock import MagicMock, patch

import numpy as np

from mpixel.datamodules.sib200 import (
    SIB200DataModule,
    SIB200Dataset,
    _make_renderer,
    preprocess_text_for_classification,
)


def test_preprocess_text_for_classification_continuous(mock_pixel_processor, mock_transforms):
    result = preprocess_text_for_classification(
        text="hello world",
        pixel_processor=mock_pixel_processor,
        transforms=mock_transforms,
        patch_size=24,
        max_num_patches=8,
        font_size=16,
    )

    assert set(result.keys()) == {"pixel_values", "attention_mask", "num_text_patches"}
    assert result["pixel_values"].shape == (3, 24, 8 * 24)
    assert result["attention_mask"].shape == (8,)
    assert result["attention_mask"][:result["num_text_patches"] + 1].sum() == result["num_text_patches"] + 1
    mock_pixel_processor.render_text_image.assert_called_once()


def test_preprocess_text_for_classification_segmented_uses_explicit_sep(mock_transforms):
    class FakeSegmentedProcessor:
        def __call__(self, text, block_size, font_size):
            image = np.full((block_size, 3 * block_size, 3), 255, dtype=np.uint8)
            image[:, 2 * block_size:3 * block_size, :] = 0
            return {
                "pixel_values": image,
                "segment_start_mask": np.array([1, 0, 0], dtype=np.int64),
            }

    result = preprocess_text_for_classification(
        text="hello",
        pixel_processor=FakeSegmentedProcessor(),
        transforms=mock_transforms,
        patch_size=4,
        max_num_patches=5,
        font_size=12,
    )

    assert result["num_text_patches"] == 2
    assert result["attention_mask"].tolist() == [1, 1, 1, 0, 0]
    assert result["pixel_values"][:, :, :8].sum() > 0
    assert result["pixel_values"][:, :, 8:12].sum() == 0


def test_make_renderer_accepts_all_sib200_renderer_types():
    with (
        patch("mpixel.datamodules.sib200.PixelRendererProcessor") as continuous,
        patch("mpixel.datamodules.sib200.BigramsRenderingProcessor") as bigrams,
        patch("mpixel.datamodules.sib200.AdaptiveRenderingProcessor") as adaptive,
        patch("mpixel.datamodules.sib200.WordsRenderingProcessor") as words,
    ):
        _make_renderer("continuous", font="font")
        _make_renderer("bigrams", font="font")
        _make_renderer("adaptive", font="font")
        _make_renderer("words", font="font")

    continuous.assert_called_once_with(font="font")
    bigrams.assert_called_once_with(font="font")
    adaptive.assert_called_once_with(font="font")
    words.assert_called_once_with(font="font")


@patch("mpixel.datamodules.sib200.FontConfig")
@patch("mpixel.datamodules.sib200.FONTS_NOTO_SANS_EXTENDED", new=[])
@patch("mpixel.datamodules.sib200.BigramsRenderingProcessor")
@patch("mpixel.datamodules.sib200.get_transforms")
def test_sib200_dataset_constructs_selected_renderer(mock_transforms, mock_bigrams, mock_font_config):
    mock_font_config.return_value = MagicMock()

    SIB200Dataset(
        data=[],
        input_text_key="text",
        label2id={},
        patch_size=24,
        max_num_patches=64,
        font_size=16,
        renderer_type="bigrams",
    )

    mock_bigrams.assert_called_once()


def test_sib200_datamodule_passes_renderer_type_to_dataset():
    dm = SIB200DataModule(language="eng_Latn", renderer_type="adaptive")
    dm.label2id = {"science": 0}

    with patch("mpixel.datamodules.sib200.SIB200Dataset") as dataset_cls:
        dm.setup_split([])

    assert dataset_cls.call_args.kwargs["renderer_type"] == "adaptive"
