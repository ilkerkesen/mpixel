"""Tests for scripts/utils/convert_to_huggingface.py — Lightning checkpoint to Hugging Face round trip."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForPreTraining

import mpixel.models.auto_register  # noqa: F401
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.processing import MPIXELProcessor

PATCH_SIZE = 16
MAX_NUM_PATCHES = 8
FONT_SIZE = 12
TEXTS = ["Hello world!", "Merhaba dünya!"]


@pytest.fixture(scope="module")
def convert_module():
    """Import the conversion script as a module (it's not a package)."""
    script_path = str(Path(__file__).parents[2] / "scripts" / "utils" / "convert_to_huggingface.py")
    spec = importlib.util.spec_from_file_location("convert_to_huggingface", script_path)
    assert spec is not None and spec.loader is not None, f"Could not find module at {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def converted(tmp_path, monkeypatch, convert_module):
    """Save a tiny model as a Lightning checkpoint, convert it, and return (model, output_dir)."""
    torch.manual_seed(0)
    config = MPIXELConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        decoder_hidden_size=32,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=64,
        patch_size=PATCH_SIZE,
        max_num_patches=MAX_NUM_PATCHES,
        mask_ratio=0.25,
        attn_implementation="sdpa",
    )
    model = MPIXELForPreTraining._from_config(config).eval()

    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "model_config": model.config.to_dict(),
            "hyper_parameters": {},
        },
        checkpoint_path,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": {
        "renderer_name": "adaptive",
        "patch_size": PATCH_SIZE,
        "max_num_patches": MAX_NUM_PATCHES,
        "font_size": FONT_SIZE,
    }}))
    output_dir = tmp_path / "hf"

    monkeypatch.setattr(sys, "argv", [
        "convert_to_huggingface.py",
        "--checkpoint", str(checkpoint_path),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
    ])
    convert_module.main()
    return model, output_dir


class TestRoundTrip:
    """The converted directory loads with from_pretrained and gives the same model."""

    def test_weights(self, converted):
        """AutoModelForPreTraining loads every weight unchanged."""
        model, output_dir = converted
        loaded = AutoModelForPreTraining.from_pretrained(output_dir)
        assert isinstance(loaded, MPIXELForPreTraining)
        expected, actual = model.state_dict(), loaded.state_dict()
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert torch.equal(actual[name], value), name

    def test_processor_config(self, converted):
        """The processor keeps the rendering settings of the training config."""
        _, output_dir = converted
        processor = MPIXELProcessor.from_pretrained(output_dir)
        assert processor.to_dict() == {
            "patch_size": PATCH_SIZE,
            "max_num_patches": MAX_NUM_PATCHES,
            "font_size": FONT_SIZE,
        }

    def test_outputs(self, converted):
        """On processor inputs, the loaded model gives the same hidden states, logits and loss."""
        model, output_dir = converted
        loaded = AutoModelForPreTraining.from_pretrained(output_dir).eval()
        inputs = MPIXELProcessor.from_pretrained(output_dir)(TEXTS)
        patch_mask = torch.zeros(len(TEXTS), MAX_NUM_PATCHES)
        patch_mask[:, 1:3] = 1

        outputs = []
        for m in (model, loaded):
            with torch.no_grad():
                encoder = m.vit(**inputs, output_hidden_states=True)
                torch.manual_seed(1)  # controlled masking adds a small random noise
                out = m(**inputs, patch_mask=patch_mask)
            outputs.append((encoder, out))
        (expected_enc, expected_out), (actual_enc, actual_out) = outputs

        exact = dict(rtol=0, atol=0)
        torch.testing.assert_close(actual_enc.last_hidden_state, expected_enc.last_hidden_state, **exact)
        assert len(actual_enc.hidden_states) == len(expected_enc.hidden_states)
        for actual_h, expected_h in zip(actual_enc.hidden_states, expected_enc.hidden_states):
            torch.testing.assert_close(actual_h, expected_h, **exact)
        torch.testing.assert_close(actual_out.logits, expected_out.logits, **exact)
        torch.testing.assert_close(actual_out.loss, expected_out.loss, **exact)
