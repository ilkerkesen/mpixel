"""Tests for mpixel/experiments/ner.py — the Lightning NER experiment."""

from copy import deepcopy

import lightning as L
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from mpixel.datamodules.ner import NERDataModule
from mpixel.experiments.ner import NERExperiment
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.utils.checkpoint import load_model_from_checkpoint

DATASET = "universal-ner-en_ewt"
MAX_NUM_PATCHES = 16
LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
SENTENCES = [
    (["John", "lives", "in", "Paris"], [1, 0, 0, 3]),
    (["Ada", "Lovelace", "wrote"], [1, 2, 0]),
]


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


def save_ner_dataset(data_dir):
    """Save a tiny NER dataset in the format of the scripts/downstream scripts."""
    features = Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(names=LABELS))})
    tokens, tags = zip(*SENTENCES)
    split = Dataset.from_dict({"tokens": list(tokens), "ner_tags": list(tags)}, features=features)
    DatasetDict(train=split, validation=split).save_to_disk(str(data_dir / DATASET))


def test_checkpoint_has_conversion_settings(tmp_path):
    """A fine-tuning checkpoint stores the model config with the label names, and the rendering settings. Then a
    conversion to an HF model needs no other files."""
    pretraining_config = save_pretraining_checkpoint(tmp_path / "pretraining.ckpt")
    save_ner_dataset(tmp_path / "data")
    datamodule = NERDataModule(
        dataset=DATASET, root_data_dir=str(tmp_path / "data"), batch_size=2, num_workers=0,
        max_num_patches=MAX_NUM_PATCHES,
    )
    experiment = NERExperiment(
        model_id_or_path=str(tmp_path / "pretraining.ckpt"), labels=datamodule.labels, dropout=0.2,
        max_num_patches=MAX_NUM_PATCHES,
    )
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_steps=1, limit_val_batches=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path,
    )
    trainer.fit(experiment, datamodule=datamodule)
    trainer.save_checkpoint(tmp_path / "finetuned.ckpt")
    checkpoint = torch.load(tmp_path / "finetuned.ckpt", map_location="cpu", weights_only=False)

    expected = deepcopy(pretraining_config)
    expected.id2label = dict(enumerate(LABELS))
    expected.label2id = {label: i for i, label in enumerate(LABELS)}
    expected.hidden_dropout_prob = expected.attention_probs_dropout_prob = 0.2
    expected.reference_compile = False  # MPIXELModel.forward changes None to False (_maybe_set_compile)
    assert checkpoint["model_config"] == expected.to_dict()
    assert checkpoint["processor_config"] == {
        "patch_size": 16, "font_size": 12, "max_num_patches": MAX_NUM_PATCHES, "renderer_type": "adaptive",
    }


def test_hf_model_folder(tmp_path):
    """An HF model folder gives the same model as its Lightning checkpoint: the same encoder weights, and with the
    same seed the same head. reference_compile is on, the same as in the checkpoints of the reported runs."""
    save_pretraining_checkpoint(tmp_path / "pretraining.ckpt")
    save_hf_model(tmp_path / "pretraining.ckpt", tmp_path / "hf")
    experiments = {}
    for name in ("pretraining.ckpt", "hf"):
        torch.manual_seed(0)
        experiments[name] = NERExperiment(
            model_id_or_path=str(tmp_path / name), labels=LABELS, dropout=0.2, max_num_patches=MAX_NUM_PATCHES,
        )
    expected, actual = experiments["pretraining.ckpt"].model.state_dict(), experiments["hf"].model.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name

    config = experiments["hf"].model_config
    assert (config.reference_compile, config.num_labels, config.hidden_dropout_prob) == (True, len(LABELS), 0.2)
