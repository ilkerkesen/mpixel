import numpy as np

from mpixel.datamodules import udp as udp_mod
from mpixel.datamodules.udp import (
    UDDataset,
    UDInputExample,
    _infer_text_direction,
    _remap_labels_to_visual_order,
    _word_order_for_direction,
)
from mpixel.utils import Modality


def test_infer_text_direction_from_treebank_language():
    assert _infer_text_direction("/data/UD_Arabic-PADT") == "rtl"
    assert _infer_text_direction("/data/UD_Hebrew-HTB") == "rtl"
    assert _infer_text_direction("/data/UD_Persian-Seraji") == "rtl"
    assert _infer_text_direction("/data/UD_Urdu-UDTB") == "rtl"
    assert _infer_text_direction("/data/UD_English-EWT") == "ltr"


def test_word_order_for_rtl_reverses_sentence():
    assert _word_order_for_direction(3, "rtl") == [2, 1, 0]


def test_word_order_for_ltr_keeps_sentence():
    assert _word_order_for_direction(3, "ltr") == [0, 1, 2]


def test_word_order_for_rtl_ignores_mixed_spans():
    # UDP direction is treebank-level. If the treebank is RTL, all UD tokens
    # reverse even if one token happens to be Latin.
    words = ["hello", "שלום", "עולם"]
    assert [words[i] for i in _word_order_for_direction(len(words), "rtl")] == ["עולם", "שלום", "hello"]


def test_remap_labels_to_visual_order():
    example = UDInputExample(
        words=["אני", "אוהב", "עברית"],
        head_labels=[3, 3, 0],
        rel_labels=["dep", "dep", "root"],
    )
    arc_labels, rel_labels = _remap_labels_to_visual_order(
        example,
        visual_to_logical=[2, 1, 0],
        n_words=3,
        label_map={"dep": 0, "root": 1},
        pad_token=-100,
    )

    assert arc_labels == [0, 1, 1]
    assert rel_labels == [1, 0, 0]


def test_remap_labels_pads_heads_dropped_by_truncation():
    example = UDInputExample(
        words=["אני", "אוהב", "עברית"],
        head_labels=[3, 1, 0],
        rel_labels=["dep", "dep", "root"],
    )
    arc_labels, rel_labels = _remap_labels_to_visual_order(
        example,
        visual_to_logical=[2, 1, 0],
        n_words=2,
        label_map={"dep": 0, "root": 1},
        pad_token=-100,
    )

    assert arc_labels == [0, -100]
    assert rel_labels == [1, 0]


def test_process_image_renders_visual_order_and_remapped_labels(monkeypatch):
    rendered_units = []

    def fake_render_units(units, block_size, font_size, pad):
        rendered_units.extend(units)
        image = np.full((block_size, len(units) * block_size, 3), 255, dtype=np.uint8)
        return image, [1] * len(units)

    monkeypatch.setattr(udp_mod, "_render_units", fake_render_units)

    example = UDInputExample(
        words=["אני", "אוהב", "עברית"],
        head_labels=[3, 3, 0],
        rel_labels=["dep", "dep", "root"],
    )
    dataset = UDDataset(
        examples=[example],
        label_list=["dep", "root"],
        modality=Modality.IMAGE,
        max_num_patches=8,
        patch_size=4,
        font_size=12,
        renderer_type="words",
        text_direction="rtl",
    )
    dataset.renderer._font_owner = None

    item = dataset._process_image(example)

    assert rendered_units == ["עברית", "אוהב", "אני"]
    assert item["word_starts"][:4].tolist() == [0, 1, 2, 3]
    assert item["arc_labels"][:3].tolist() == [0, 1, 1]
    assert item["rel_labels"][:3].tolist() == [1, 0, 0]


def test_process_image_renderer_type_bigrams_controls_units(monkeypatch):
    rendered_units = []

    def fake_render_units(units, block_size, font_size, pad):
        rendered_units.extend(units)
        image = np.full((block_size, len(units) * block_size, 3), 255, dtype=np.uint8)
        return image, [1] * len(units)

    monkeypatch.setattr(udp_mod, "_render_units", fake_render_units)

    example = UDInputExample(
        words=["hello", "world"],
        head_labels=[2, 0],
        rel_labels=["dep", "root"],
    )
    dataset = UDDataset(
        examples=[example],
        label_list=["dep", "root"],
        modality=Modality.IMAGE,
        max_num_patches=16,
        patch_size=4,
        font_size=12,
        renderer_type="bigrams",
    )
    dataset.renderer._font_owner = None

    item = dataset._process_image(example)

    assert rendered_units == ["he", "ll", "o", "wo", "rl", "d"]
    assert item["word_starts"][:3].tolist() == [0, 3, 6]
    assert item["arc_labels"][:2].tolist() == [2, 0]


def test_process_image_renderer_type_adaptive_uses_script_dispatch(monkeypatch):
    rendered_units = []

    def fake_render_units(units, block_size, font_size, pad):
        rendered_units.extend(units)
        image = np.full((block_size, len(units) * block_size, 3), 255, dtype=np.uint8)
        return image, [1] * len(units)

    monkeypatch.setattr(udp_mod, "_render_units", fake_render_units)

    example = UDInputExample(
        words=["hello", "مرحبا"],
        head_labels=[2, 0],
        rel_labels=["dep", "root"],
    )
    dataset = UDDataset(
        examples=[example],
        label_list=["dep", "root"],
        modality=Modality.IMAGE,
        max_num_patches=16,
        patch_size=4,
        font_size=12,
        renderer_type="adaptive",
    )
    dataset.renderer._font_owner = None

    item = dataset._process_image(example)

    assert rendered_units == ["he", "ll", "o", "مرحبا"]
    assert item["word_starts"][:3].tolist() == [0, 3, 4]
    assert item["arc_labels"][:2].tolist() == [2, 0]
