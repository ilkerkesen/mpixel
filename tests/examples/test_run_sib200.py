"""Tests for examples/run_sib200.py — fine-tune MPIXELForSequenceClassification on SIB-200 with the HF Trainer."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, Trainer

import mpixel.models.auto_register  # noqa: F401
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.processing import MPIXELProcessor

LANGUAGE = "eng_Latn"
MAX_NUM_PATCHES = 8
# The processor that scripts/utils/convert_to_huggingface.py saves for the released models.
CHECKPOINT_PROCESSOR = dict(patch_size=16, max_num_patches=512, font_size=12)
# The seven SIB-200 categories, in sorted order (the label ids of SIB200DataModule).
CATEGORIES = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]


@pytest.fixture(scope="module")
def run_sib200():
    """Import the example script as a module (it's not a package)."""
    script_path = str(Path(__file__).parents[2] / "examples" / "run_sib200.py")
    spec = importlib.util.spec_from_file_location("run_sib200", script_path)
    assert spec is not None and spec.loader is not None, f"Could not find module at {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_converted_model(model_dir: Path) -> Path:
    """Save a tiny MPIXELForPreTraining and its processor, the same as scripts/utils/convert_to_huggingface.py."""
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
        attn_implementation="sdpa",
    )
    MPIXELForPreTraining._from_config(config).save_pretrained(model_dir)
    MPIXELProcessor(**CHECKPOINT_PROCESSOR).save_pretrained(model_dir)
    return model_dir


def save_sib200_data(data_dir: Path) -> Path:
    """Save a tiny SIB-200 dataset. SIB200DataModule.prepare_data skips the download when the folder exists."""
    split = Dataset.from_dict({"text": [f"A sentence about {c}." for c in CATEGORIES], "category": CATEGORIES})
    DatasetDict(train=split, validation=split, test=split).save_to_disk(str(data_dir / "sib-200" / LANGUAGE))
    return data_dir


def short_run_args(model_dir: Path, data_dir: Path, output_dir: Path, *extra: str) -> list:
    """Options for a short run on the CPU."""
    return [
        "--model_name_or_path", str(model_dir),
        "--language", LANGUAGE,
        "--root_data_dir", str(data_dir),
        "--max_num_patches", str(MAX_NUM_PATCHES),
        "--output_dir", str(output_dir),
        "--max_steps", "2",
        "--per_device_train_batch_size", "4",
        "--per_device_eval_batch_size", "4",
        "--bf16", "False",
        "--use_cpu",
        "--dataloader_num_workers", "0",
        "--report_to", "none",
        *extra,
    ]


def test_lightning_defaults(run_sib200):
    """Without options, the script uses the settings of the Lightning SIB-200 runs, with seed 0."""
    model_args, data_args, early_stopping_args, training_args = run_sib200.parse_args(
        ["--model_name_or_path", "model", "--language", LANGUAGE, "--output_dir", "out", "--use_cpu"]
    )
    # The head settings come from the checkpoint (see test_train), the dropout is 0.1, and the head gets the
    # PyTorch default init, the same as the Lightning runs
    assert (model_args.pooling_mode, model_args.pooler_add_layer_norm, model_args.dropout) == (None, None, 0.1)
    assert model_args.head_init == "pytorch"
    assert model_args.reference_compile is False
    assert data_args.max_num_patches == 256
    assert early_stopping_args.early_stopping_patience == 20
    expected = {
        "optim": "adamw_torch",
        "learning_rate": 5e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "max_steps": 15_000,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "eval_strategy": "epoch",
        "save_strategy": "best",  # not at each epoch: a SIB-200 epoch has only about 11 steps
        "save_total_limit": 2,
        "dataloader_persistent_workers": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": 0,
        "remove_unused_columns": False,  # the transform needs the text and category columns
        "do_train": True,  # train, then validate and test
        "do_eval": True,
        "do_predict": True,
    }
    assert {key: getattr(training_args, key) for key in expected} == expected


def test_json_args(run_sib200, tmp_path):
    """One JSON file gives the same options as the command line, with the same defaults."""
    options = {"model_name_or_path": "model", "language": LANGUAGE, "output_dir": "out", "learning_rate": 1e-4}
    json_path = tmp_path / "args.json"
    json_path.write_text(json.dumps({**options, "use_cpu": True}))
    command_line = [arg for key, value in options.items() for arg in (f"--{key}", str(value))] + ["--use_cpu"]

    *json_args, json_training_args = run_sib200.parse_args([str(json_path)])
    *cli_args, cli_training_args = run_sib200.parse_args(command_line)
    assert json_args == cli_args
    # logging_dir has the time in its default value
    json_training, cli_training = json_training_args.to_dict(), cli_training_args.to_dict()
    del json_training["logging_dir"], cli_training["logging_dir"]
    assert json_training == cli_training
    assert (json_training_args.optim, json_training_args.learning_rate) == ("adamw_torch", 1e-4)


def test_get_processor(run_sib200, tmp_path):
    """The script renders with the MPIXELProcessor of the checkpoint, with the SIB-200 number of patches."""
    MPIXELProcessor(**CHECKPOINT_PROCESSOR).save_pretrained(tmp_path)
    processor = run_sib200.get_processor(str(tmp_path), max_num_patches=256)
    assert isinstance(processor, MPIXELProcessor)
    assert processor.to_dict() == {**CHECKPOINT_PROCESSOR, "max_num_patches": 256}


def test_train(run_sib200, tmp_path):
    """A short run trains with the --dropout and --head_init values, saves the best model, and writes the validation
    and test results."""
    model_dir, data_dir = save_converted_model(tmp_path / "model"), save_sib200_data(tmp_path / "data")
    output_dir = tmp_path / "out"
    run_sib200.main(short_run_args(model_dir, data_dir, output_dir, "--dropout", "0.2", "--head_init", "normal"))

    results = json.loads((output_dir / "all_results.json").read_text())
    for key in ("train_loss", "eval_loss", "eval_accuracy", "eval_f1", "test_loss", "test_accuracy", "test_f1"):
        assert key in results, key
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    assert model.config.id2label == dict(enumerate(CATEGORIES))
    assert (model.config.hidden_dropout_prob, model.config.attention_probs_dropout_prob) == (0.2, 0.2)
    assert model.config.head_init == "normal"
    # The head settings of the converted checkpoint are the Lightning defaults
    assert (model.config.pooling_mode, model.config.pooler_add_layer_norm) == ("mean", True)


def test_evaluate_only(run_sib200, tmp_path):
    """With --do_train False, the script evaluates a fine-tuned model with the head settings of its run, and gets
    the same results as at the end of that run."""
    model_dir, data_dir = save_converted_model(tmp_path / "model"), save_sib200_data(tmp_path / "data")
    run_sib200.main(short_run_args(model_dir, data_dir, tmp_path / "train", "--pooling_mode", "cls"))
    run_sib200.main(short_run_args(tmp_path / "train", data_dir, tmp_path / "eval", "--do_train", "False"))

    trained = json.loads((tmp_path / "train" / "all_results.json").read_text())
    evaluated = json.loads((tmp_path / "eval" / "all_results.json").read_text())
    assert not any(key.startswith("train_") for key in evaluated)
    for split in ("eval", "test"):
        for metric in ("loss", "accuracy", "f1"):
            key = f"{split}_{metric}"
            assert evaluated[key] == trained[key], key


def test_resume(run_sib200, tmp_path, monkeypatch):
    """If output_dir has a checkpoint, the script continues from the last one, and gets the same weights as a run
    without a stop."""
    model_dir, data_dir = save_converted_model(tmp_path / "model"), save_sib200_data(tmp_path / "data")
    output_dir = tmp_path / "out"
    args = short_run_args(
        model_dir, data_dir, output_dir,
        "--eval_strategy", "steps", "--eval_steps", "1", "--save_strategy", "steps", "--save_steps", "1",
    )
    run_sib200.main(args)
    expected = load_file(output_dir / "checkpoint-2" / "model.safetensors")
    shutil.rmtree(output_dir / "checkpoint-2")  # as if the run stopped after step 1

    resumed_from = []
    train = Trainer.train

    def record_resume(self, *train_args, **train_kwargs):
        resumed_from.append(train_kwargs.get("resume_from_checkpoint"))
        return train(self, *train_args, **train_kwargs)

    monkeypatch.setattr(Trainer, "train", record_resume)
    run_sib200.main(args)

    assert resumed_from == [str(output_dir / "checkpoint-1")]
    actual = load_file(output_dir / "checkpoint-2" / "model.safetensors")
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_output_dir_not_empty(run_sib200, tmp_path):
    """If output_dir has files but no checkpoint, the script stops, unless --overwrite_output_dir is given."""
    (tmp_path / "notes.txt").write_text("")

    def training_args(*extra):
        command_line = ["--model_name_or_path", "model", "--language", LANGUAGE, "--output_dir", str(tmp_path)]
        return run_sib200.parse_args([*command_line, "--use_cpu", *extra])[-1]

    with pytest.raises(ValueError, match="not empty"):
        run_sib200.get_resume_checkpoint(training_args())
    assert run_sib200.get_resume_checkpoint(training_args("--overwrite_output_dir")) is None
