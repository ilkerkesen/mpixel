"""Tests for the MPIXEL task models in mpixel/models/mpixel/modeling_mpixel.py."""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

import mpixel.models.auto_register  # noqa: F401
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.processing import MPIXELProcessor

PATCH_SIZE = 16
MAX_NUM_PATCHES = 8
FONT_SIZE = 12
TEXTS = ["Hello world!", "Merhaba dünya!"]
NUM_LABELS = 3


@pytest.fixture
def pretrained(tmp_path):
    """Save a tiny MPIXELForPreTraining and its processor as a converted HF checkpoint; return (model, dir)."""
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
        attn_implementation="sdpa",
    )
    model = MPIXELForPreTraining._from_config(config).eval()
    # scripts/utils/convert_to_huggingface.py writes its output with these two calls.
    model.save_pretrained(tmp_path)
    MPIXELProcessor(patch_size=PATCH_SIZE, max_num_patches=MAX_NUM_PATCHES, font_size=FONT_SIZE).save_pretrained(tmp_path)
    return model, tmp_path


class TestSequenceClassification:
    """A converted checkpoint loads with AutoModelForSequenceClassification."""

    def test_from_pretrained(self, pretrained):
        """The encoder weights load unchanged, and a forward pass gives logits and a loss."""
        model, path = pretrained
        loaded = AutoModelForSequenceClassification.from_pretrained(path, num_labels=NUM_LABELS).eval()
        expected, actual = model.vit.state_dict(), loaded.vit.state_dict()
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert torch.equal(actual[name], value), name

        inputs = MPIXELProcessor.from_pretrained(path)(TEXTS)
        labels = torch.tensor([0, NUM_LABELS - 1])
        with torch.no_grad():
            outputs = loaded(**inputs, labels=labels)
        assert outputs.logits.shape == (len(TEXTS), NUM_LABELS)
        assert outputs.loss.ndim == 0
        assert torch.isfinite(outputs.loss)

    def test_head_init(self, pretrained):
        """The new head gets the PyTorch default init, the same as the Lightning fine-tuning code."""
        _, path = pretrained
        loaded = AutoModelForSequenceClassification.from_pretrained(path, num_labels=NUM_LABELS)
        for linear in (loaded.pooler.linear, loaded.classifier):
            # nn.Linear.reset_parameters: weight and bias from U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            bound = linear.in_features**-0.5
            for param in (linear.weight, linear.bias):
                assert param.abs().max() <= bound
            assert linear.bias.abs().sum() > 0
            torch.testing.assert_close(linear.weight.std(), torch.tensor(bound / 3**0.5), rtol=0.25, atol=0)

    def test_head_init_normal(self, pretrained):
        """With head_init="normal", the new head gets the HF default init: normal weights and zero biases."""
        _, path = pretrained
        loaded = AutoModelForSequenceClassification.from_pretrained(path, num_labels=NUM_LABELS, head_init="normal")
        for linear in (loaded.pooler.linear, loaded.classifier):
            assert torch.equal(linear.bias, torch.zeros_like(linear.bias))
            std = torch.tensor(loaded.config.initializer_range)
            torch.testing.assert_close(linear.weight.std(), std, rtol=0.25, atol=0)
        assert torch.equal(loaded.pooler.ln.weight, torch.ones_like(loaded.pooler.ln.weight))
        assert torch.equal(loaded.pooler.ln.bias, torch.zeros_like(loaded.pooler.ln.bias))


class TestTokenClassification:
    """A converted checkpoint loads with AutoModelForTokenClassification."""

    def test_from_pretrained(self, pretrained):
        """The encoder weights load unchanged, and a forward pass gives per-patch logits and a loss."""
        model, path = pretrained
        loaded = AutoModelForTokenClassification.from_pretrained(path, num_labels=NUM_LABELS).eval()
        expected, actual = model.vit.state_dict(), loaded.vit.state_dict()
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert torch.equal(actual[name], value), name

        inputs = MPIXELProcessor.from_pretrained(path)(TEXTS)
        # One label per patch, -100 on the patches without a label (the same as NERDataModule)
        labels = torch.full((len(TEXTS), MAX_NUM_PATCHES), -100)
        labels[:, 0] = torch.tensor([0, NUM_LABELS - 1])
        with torch.no_grad():
            outputs = loaded(**inputs, labels=labels)
        assert outputs.logits.shape == (len(TEXTS), MAX_NUM_PATCHES, NUM_LABELS)
        assert outputs.loss.ndim == 0
        assert torch.isfinite(outputs.loss)

    def test_head_init(self, pretrained):
        """The new head gets the PyTorch default init, the same as the Lightning fine-tuning code."""
        _, path = pretrained
        loaded = AutoModelForTokenClassification.from_pretrained(path, num_labels=NUM_LABELS)
        linear = loaded.classifier
        # nn.Linear.reset_parameters: weight and bias from U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bound = linear.in_features**-0.5
        for param in (linear.weight, linear.bias):
            assert param.abs().max() <= bound
        assert linear.bias.abs().sum() > 0
        torch.testing.assert_close(linear.weight.std(), torch.tensor(bound / 3**0.5), rtol=0.25, atol=0)
