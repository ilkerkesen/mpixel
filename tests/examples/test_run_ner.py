"""Tests for examples/run_ner.py — fine-tune MPIXELForTokenClassification on NER with the HF Trainer."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from safetensors.torch import load_file
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, EvalPrediction, Trainer

import mpixel.models.auto_register  # noqa: F401
from mpixel.datamodules.ner import NERDataModule
from mpixel.experiments.ner import NERExperiment
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining

DATASET = "universal-ner-en_ewt"
MAX_NUM_PATCHES = 16
LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
SENTENCES = [
    (["John", "lives", "in", "Paris"], [1, 0, 0, 3]),
    (["Ada", "Lovelace", "wrote"], [1, 2, 0]),
    (["New", "York", "is", "big"], [3, 4, 0, 0]),
    (["We", "met", "Mary"], [0, 0, 1]),
]


@pytest.fixture(scope="module")
def run_ner():
    """Import the example script as a module (it's not a package)."""
    script_path = str(Path(__file__).parents[2] / "examples" / "run_ner.py")
    spec = importlib.util.spec_from_file_location("run_ner", script_path)
    assert spec is not None and spec.loader is not None, f"Could not find module at {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_ner_dataset(data_dir: Path, with_test: bool = True):
    """Save a tiny NER dataset in the format of the scripts/downstream scripts: tokens and ClassLabel ner_tags."""
    features = Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(names=LABELS))})
    tokens, tags = zip(*SENTENCES)
    split = Dataset.from_dict({"tokens": list(tokens), "ner_tags": list(tags)}, features=features)
    splits = dict(train=split, validation=split, **(dict(test=split) if with_test else {}))
    DatasetDict(splits).save_to_disk(str(data_dir / DATASET))
    return data_dir


def save_converted_model(model_dir: Path) -> Path:
    """Save a tiny MPIXELForPreTraining, the same as scripts/utils/convert_to_huggingface.py."""
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
    return model_dir


def short_run_args(model_dir: Path, data_dir: Path, output_dir: Path, *extra: str) -> list:
    """Options for a short run on the CPU, with an evaluation and a checkpoint after each step."""
    return [
        "--model_name_or_path", str(model_dir),
        "--dataset", DATASET,
        "--root_data_dir", str(data_dir),
        "--max_num_patches", str(MAX_NUM_PATCHES),
        "--output_dir", str(output_dir),
        "--max_steps", "2",
        "--eval_steps", "1",
        "--save_steps", "1",
        "--per_device_train_batch_size", "2",
        "--per_device_eval_batch_size", "2",
        "--bf16", "False",
        "--use_cpu",
        "--dataloader_num_workers", "0",
        "--report_to", "none",
        *extra,
    ]


def test_lightning_defaults(run_ner):
    """Without options, the script uses the settings of the Lightning NER runs, with seed 0."""
    model_args, data_args, early_stopping_args, training_args = run_ner.parse_args(
        ["--model_name_or_path", "model", "--dataset", DATASET, "--output_dir", "out", "--use_cpu"]
    )
    assert model_args.dropout == 0.1
    assert (data_args.root_data_dir, data_args.max_num_patches) == ("./data/ner", 256)
    assert early_stopping_args.early_stopping_patience == 5
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
        "eval_strategy": "steps",  # val_check_interval=500
        "eval_steps": 500,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "dataloader_num_workers": 8,
        "seed": 0,
        "do_train": True,  # train, then validate and test
        "do_eval": True,
        "do_predict": True,
    }
    assert {key: getattr(training_args, key) for key in expected} == expected


def test_json_args(run_ner, tmp_path):
    """One JSON file gives the same options as the command line, with the same defaults."""
    options = {"model_name_or_path": "model", "dataset": DATASET, "output_dir": "out", "learning_rate": 1e-4}
    json_path = tmp_path / "args.json"
    json_path.write_text(json.dumps({**options, "use_cpu": True}))
    command_line = [arg for key, value in options.items() for arg in (f"--{key}", str(value))] + ["--use_cpu"]

    *json_args, json_training_args = run_ner.parse_args([str(json_path)])
    *cli_args, cli_training_args = run_ner.parse_args(command_line)
    assert json_args == cli_args
    # logging_dir has the time in its default value
    json_training, cli_training = json_training_args.to_dict(), cli_training_args.to_dict()
    del json_training["logging_dir"], cli_training["logging_dir"]
    assert json_training == cli_training
    assert (json_training_args.optim, json_training_args.learning_rate) == ("adamw_torch", 1e-4)


def test_get_datamodule(run_ner, tmp_path):
    """The script gets its data from NERDataModule, with the rendering settings of the Lightning NER runs."""
    save_ner_dataset(tmp_path)
    _, data_args, _, _ = run_ner.parse_args([
        "--model_name_or_path", "model", "--dataset", DATASET, "--root_data_dir", str(tmp_path),
        "--output_dir", "out", "--use_cpu",
    ])
    datamodule = run_ner.get_datamodule(data_args)
    assert isinstance(datamodule, NERDataModule)
    assert datamodule.labels == LABELS
    for dataset in (datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset):
        settings = (dataset.patch_size, dataset.font_size, dataset.renderer_type, dataset.pad_label_id)
        assert settings == (16, 12, "adaptive", -100)
        assert dataset.max_num_patches == 256


def test_compute_metrics(run_ner):
    """The metrics are the seqeval metrics of NERExperiment, on the patches that have a label."""
    torch.manual_seed(0)
    logits = torch.randn(len(SENTENCES), MAX_NUM_PATCHES, len(LABELS))
    labels = torch.full((len(SENTENCES), MAX_NUM_PATCHES), -100)
    for i, (_, tags) in enumerate(SENTENCES):
        labels[i, : 2 * len(tags) : 2] = torch.tensor(tags)  # one label on the first patch of each token

    # NERExperiment._accumulate_predictions and _log_seqeval_metrics
    lightning = SimpleNamespace(pad_label_id=-100, id2label=dict(enumerate(LABELS)), _val_preds=[], _val_golds=[])
    NERExperiment._accumulate_predictions(lightning, "val", logits, labels)
    golds, preds = lightning._val_golds, lightning._val_preds
    expected = {
        "accuracy": accuracy_score(golds, preds),
        "precision": precision_score(golds, preds),
        "recall": recall_score(golds, preds),
        "f1": f1_score(golds, preds),
    }

    predictions = run_ner.preprocess_logits_for_metrics(logits, labels)
    eval_pred = EvalPrediction(predictions=predictions.numpy(), label_ids=labels.numpy())
    assert run_ner.compute_metrics(eval_pred, id2label=dict(enumerate(LABELS))) == expected


@pytest.mark.parametrize("with_test", [True, False], ids=["with-test-split", "no-test-split"])
def test_train(run_ner, tmp_path, with_test):
    """A short run trains with the --dropout value, saves the best model, and writes the validation results.
    It writes the test results only if the dataset has a test split (klue-ner has none)."""
    model_dir = save_converted_model(tmp_path / "model")
    data_dir = save_ner_dataset(tmp_path / "data", with_test=with_test)
    output_dir = tmp_path / "out"
    run_ner.main(short_run_args(model_dir, data_dir, output_dir, "--dropout", "0.2"))

    results = json.loads((output_dir / "all_results.json").read_text())
    assert "train_loss" in results
    for split in ("eval", "test"):
        for metric in ("loss", "accuracy", "precision", "recall", "f1"):
            assert (f"{split}_{metric}" in results) == (split == "eval" or with_test), f"{split}_{metric}"
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    assert model.config.id2label == dict(enumerate(LABELS))
    assert (model.config.hidden_dropout_prob, model.config.attention_probs_dropout_prob) == (0.2, 0.2)


def test_evaluate_only(run_ner, tmp_path):
    """With --do_train False, the script evaluates a fine-tuned model, and gets the same results as at the end of
    its run."""
    model_dir, data_dir = save_converted_model(tmp_path / "model"), save_ner_dataset(tmp_path / "data")
    run_ner.main(short_run_args(model_dir, data_dir, tmp_path / "train"))
    run_ner.main(short_run_args(tmp_path / "train", data_dir, tmp_path / "eval", "--do_train", "False"))

    trained = json.loads((tmp_path / "train" / "all_results.json").read_text())
    evaluated = json.loads((tmp_path / "eval" / "all_results.json").read_text())
    assert not any(key.startswith("train_") for key in evaluated)
    for split in ("eval", "test"):
        for metric in ("loss", "accuracy", "precision", "recall", "f1"):
            key = f"{split}_{metric}"
            assert evaluated[key] == trained[key], key


def test_resume(run_ner, tmp_path, monkeypatch):
    """If output_dir has a checkpoint, the script continues from the last one, and gets the same weights as a run
    without a stop."""
    model_dir, data_dir = save_converted_model(tmp_path / "model"), save_ner_dataset(tmp_path / "data")
    output_dir = tmp_path / "out"
    args = short_run_args(model_dir, data_dir, output_dir)
    run_ner.main(args)
    expected = load_file(output_dir / "checkpoint-2" / "model.safetensors")
    shutil.rmtree(output_dir / "checkpoint-2")  # as if the run stopped after step 1

    resumed_from = []
    train = Trainer.train

    def record_resume(self, *train_args, **train_kwargs):
        resumed_from.append(train_kwargs.get("resume_from_checkpoint"))
        return train(self, *train_args, **train_kwargs)

    monkeypatch.setattr(Trainer, "train", record_resume)
    run_ner.main(args)

    assert resumed_from == [str(output_dir / "checkpoint-1")]
    actual = load_file(output_dir / "checkpoint-2" / "model.safetensors")
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_output_dir_not_empty(run_ner, tmp_path):
    """If output_dir has files but no checkpoint, the script stops, unless --overwrite_output_dir is given."""
    (tmp_path / "notes.txt").write_text("")

    def training_args(*extra):
        command_line = ["--model_name_or_path", "model", "--dataset", DATASET, "--output_dir", str(tmp_path)]
        return run_ner.parse_args([*command_line, "--use_cpu", *extra])[-1]

    with pytest.raises(ValueError, match="not empty"):
        run_ner.get_resume_checkpoint(training_args())
    assert run_ner.get_resume_checkpoint(training_args("--overwrite_output_dir")) is None
