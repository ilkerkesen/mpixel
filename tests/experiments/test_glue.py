"""Tests for mpixel/experiments/glue.py — the Lightning GLUE experiment."""

import torch

from mpixel.experiments.glue import GLUEExperiment
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.utils.checkpoint import load_model_from_checkpoint

MAX_NUM_PATCHES = 16


def save_pretraining_checkpoint(path) -> MPIXELConfig:
    """Save a tiny checkpoint the same as PretrainingExperiment: the weights and the model config."""
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
        max_num_patches=MAX_NUM_PATCHES,
    )
    model = MPIXELForPreTraining._from_config(config)
    state_dict = {f"model.{name}": value for name, value in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "model_config": config.to_dict()}, path)
    return config


def save_hf_model(checkpoint_path, model_dir):
    """Convert a pretraining checkpoint to an HF model folder, the same as scripts/utils/convert_to_huggingface.py."""
    model, _, _ = load_model_from_checkpoint(str(checkpoint_path), strict=True)
    model.save_pretrained(model_dir)


def test_hf_model_folder(tmp_path):
    """An HF model folder gives the same model as its Lightning checkpoint: the same encoder weights, and with the
    same seed the same head. reference_compile is on, the same as in the checkpoints of the reported runs."""
    save_pretraining_checkpoint(tmp_path / "pretraining.ckpt")
    save_hf_model(tmp_path / "pretraining.ckpt", tmp_path / "hf")
    experiments = {}
    for name in ("pretraining.ckpt", "hf"):
        torch.manual_seed(0)
        experiments[name] = GLUEExperiment(
            model_id_or_path=str(tmp_path / name), task_name="sst2", num_labels=2, dropout=0.2, max_num_patches=MAX_NUM_PATCHES,
        )
    expected, actual = experiments["pretraining.ckpt"].model.state_dict(), experiments["hf"].model.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name

    config = experiments["hf"].model_config
    assert (config.reference_compile, config.num_labels, config.hidden_dropout_prob) == (True, 2, 0.2)
