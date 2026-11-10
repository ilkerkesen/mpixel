#!/usr/bin/env python
"""
Fine-tune a converted MPIXEL checkpoint on one SIB-200 language with the HuggingFace Trainer.

This script is the Trainer version of the Lightning experiment `mpixel/experiments/text_classification.py`.
Its defaults are the defaults of that experiment, so that a run uses the same setup as the reported runs.
The text is rendered with the `MPIXELProcessor` of the checkpoint. `tests/test_processing.py` checks that
it gives the same pixels as `SIB200DataModule`.

Run it on one GPU, the same as the Lightning runs:

    CUDA_VISIBLE_DEVICES=0 python scripts/hf-examples/run_sib200.py \
        --model_name_or_path <converted-checkpoint-dir> \
        --language eng_Latn \
        --output_dir runs/sib-200/eng_Latn/seed0 \
        --seed 0

Make the checkpoint with `scripts/utils/convert_to_huggingface.py`. All `TrainingArguments` options are
available, for example `--learning_rate` or `--max_steps`. You can also give all options in one JSON file:

    python scripts/hf-examples/run_sib200.py args.json

By default, the script trains, then evaluates the best checkpoint on the validation and test splits. To only
evaluate a fine-tuned model, give its folder and add `--do_train False`. If `--output_dir` has a checkpoint, the
script continues training from the last checkpoint. To start again from the beginning, add `--overwrite_output_dir`.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME

import mpixel.models.auto_register  # noqa: F401
from mpixel.datamodules.sib200 import SIB200DataModule
from mpixel.processing import MPIXELProcessor
from mpixel.utils import get_data_dir


logger = logging.getLogger(__name__)

# Settings of the Lightning SIB-200 runs that are different from the TrainingArguments defaults. These
# settings are already the same in both: AdamW parameter groups, learning rate, betas, epsilon, weight
# decay, linear schedule without warmup, and gradient clipping at norm 1.0.
LIGHTNING_DEFAULTS = dict(
    optim="adamw_torch",  # torch.optim.AdamW, not the fused version
    bf16=True,  # precision "bf16-mixed"
    max_steps=15_000,
    per_device_train_batch_size=32,  # SIB200DataModule batch_size
    per_device_eval_batch_size=32,
    eval_strategy="epoch",  # validation once per epoch
    # Save only when eval_f1 is better, as ModelCheckpoint does. A save at each epoch is very slow, because a
    # SIB-200 epoch has only about 11 steps.
    save_strategy="best",
    save_total_limit=2,  # ModelCheckpoint save_top_k=2
    # SIB200DataModule keeps the workers between epochs. Without this, 18 new workers start at each epoch.
    dataloader_persistent_workers=True,
    load_best_model_at_end=True,  # validate and test the best checkpoint
    metric_for_best_model="f1",  # ModelCheckpoint monitor="val_f1", mode="max"
    greater_is_better=True,
    dataloader_num_workers=8,  # SIB200DataModule num_workers
    logging_steps=50,  # Trainer log_every_n_steps
    report_to="tensorboard",  # TensorBoardLogger
    seed=0,
    do_train=True,  # train, then validate and test the best checkpoint
    do_eval=True,
    do_predict=True,
)

# remove_unused_columns=False is not a Lightning setting: the transform needs the text and category columns.
SCRIPT_DEFAULTS = {**LIGHTNING_DEFAULTS, "remove_unused_columns": False}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to a converted MPIXEL checkpoint (scripts/utils/convert_to_huggingface.py), or to a "
            "model that this script fine-tuned."
        }
    )
    pooling_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Pooling of the patch embeddings: "mean", "max", "cls" or "pma<num_heads>". Default: the value '
            'of the checkpoint ("mean", the Lightning default, in a converted checkpoint).'
        },
    )
    pooler_add_layer_norm: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether the pooler applies a layer norm before pooling. Default: the value of the checkpoint "
            "(True, the Lightning default, in a converted checkpoint)."
        },
    )
    dropout: float = field(default=0.1, metadata={"help": "Hidden and attention dropout."})
    head_init: str = field(
        default="pytorch",
        metadata={
            "help": 'Init of the new pooler and classifier: "pytorch" (uniform, the Lightning runs) or "normal" (the '
            "HF default: normal weights with std initializer_range, and zero biases). A fine-tuned model loads its "
            "head, so this has no effect on it.",
            "choices": ["pytorch", "normal"],
        },
    )
    reference_compile: bool = field(
        default=False,
        metadata={
            "help": "Run the MLP of each layer and the embeddings with torch.compile. The Lightning runs did this, "
            "because the pretraining checkpoint has reference_compile=True. The outputs are the same up to float "
            "rounding."
        },
    )


@dataclass
class DataArguments:
    language: str = field(metadata={"help": "SIB-200 language config, for example eng_Latn."})
    root_data_dir: str = field(
        default=get_data_dir(), metadata={"help": "The dataset is stored in <root_data_dir>/sib-200/<language>."}
    )
    max_num_patches: int = field(
        default=256,  # SIB200DataModule max_num_patches
        metadata={"help": "Maximum number of patches of a rendered text. Replaces the value of the checkpoint."},
    )


@dataclass
class EarlyStoppingArguments:
    early_stopping_patience: int = field(
        default=20, metadata={"help": "Stop when the validation macro-F1 does not improve for this many evaluations."}
    )


def parse_args(args=None):
    """Parse the command line, or one JSON file with the same options. Both use the defaults of this script."""
    parser = HfArgumentParser((ModelArguments, DataArguments, EarlyStoppingArguments, TrainingArguments))
    parser.set_defaults(**SCRIPT_DEFAULTS)
    args = sys.argv[1:] if args is None else args
    if len(args) == 1 and args[0].endswith(".json"):
        # parse_json_file does not use the parser defaults, so give them to parse_dict with the JSON values.
        with open(args[0]) as f:
            return parser.parse_dict({**SCRIPT_DEFAULTS, **json.load(f)})
    return parser.parse_args_into_dataclasses(args=args)


def get_resume_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    """The checkpoint to continue training from: --resume_from_checkpoint, else the last checkpoint in output_dir."""
    if training_args.resume_from_checkpoint is not None:
        return training_args.resume_from_checkpoint
    output_dir = training_args.output_dir
    if training_args.overwrite_output_dir or not os.path.isdir(output_dir):
        return None
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None and os.listdir(output_dir):
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to train from the beginning."
        )
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    return last_checkpoint


def get_processor(model_name_or_path: str, max_num_patches: int) -> MPIXELProcessor:
    """The processor of the checkpoint, with the number of patches of the SIB-200 runs."""
    processor = MPIXELProcessor.from_pretrained(model_name_or_path)
    processor.max_num_patches = max_num_patches
    return processor


def preprocess(batch, processor: MPIXELProcessor, label2id: dict):
    """Render a batch of texts, and give each category its label id."""
    inputs = processor(batch["text"])
    inputs["labels"] = torch.tensor([label2id[category] for category in batch["category"]])
    return inputs


def collate_fn(examples):
    """The model inputs of SIB200DataModule.collate_fn."""
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "attention_mask": torch.stack([example["attention_mask"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
    }


def compute_metrics(eval_pred: EvalPrediction):
    """Accuracy and macro-F1 with the same torchmetrics functions as the Lightning experiment."""
    logits = torch.from_numpy(eval_pred.predictions)
    labels = torch.from_numpy(eval_pred.label_ids)
    num_classes = logits.shape[-1]
    return {
        "accuracy": multiclass_accuracy(logits, labels, num_classes=num_classes, average="micro").item(),
        "f1": multiclass_f1_score(logits, labels, num_classes=num_classes, average="macro").item(),
    }


def load_best_checkpoint(trainer: Trainer) -> None:
    """Load the checkpoint with the best validation result into the model of the trainer.

    With save_strategy="best", transformers 4.57 saves a checkpoint at each new best result, but it does not
    load the best checkpoint at the end, because it does not record its step (huggingface/transformers#44583).
    The step is in the evaluation logs, so get it from there.
    """
    args = trainer.args
    metric = args.metric_for_best_model
    metric = metric if metric.startswith("eval_") else f"eval_{metric}"
    evaluations = [log for log in trainer.state.log_history if metric in log]
    if not evaluations:
        return
    # max and min give the first of equal results, the same as the Trainer: only a better result is a new best.
    best = (max if args.greater_is_better else min)(evaluations, key=lambda log: log[metric])
    checkpoint = os.path.join(args.output_dir, f"checkpoint-{best['step']}")
    logger.info(f"Loading the best model from {checkpoint} ({metric} = {best[metric]}).")
    trainer.model.load_state_dict(load_file(os.path.join(checkpoint, SAFE_WEIGHTS_NAME)))


def main(args=None):
    model_args, data_args, early_stopping_args, training_args = parse_args(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    if training_args.n_gpu > 1 or training_args.world_size > 1:
        raise ValueError(
            "The Lightning SIB-200 runs used one GPU. More GPUs change the batch size. "
            "Run on one GPU, for example with CUDA_VISIBLE_DEVICES=0."
        )
    resume_checkpoint = get_resume_checkpoint(training_args) if training_args.do_train else None
    if training_args.dataloader_num_workers == 0:
        # PyTorch does not accept persistent workers without workers (SIB200DataModule: num_workers > 0).
        training_args.dataloader_persistent_workers = False

    # Download the dataset once to <root_data_dir>/sib-200/<language>, the same folder as the Lightning runs.
    datamodule = SIB200DataModule(language=data_args.language, root_data_dir=data_args.root_data_dir)
    datamodule.prepare_data()
    raw_datasets = load_from_disk(datamodule.dataset_dir)

    # The same label ids as SIB200DataModule.setup: the sorted categories of the training split.
    labels = sorted(raw_datasets["train"].unique("category"))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = dict(enumerate(labels))

    processor = get_processor(model_args.model_name_or_path, data_args.max_num_patches)
    raw_datasets.set_transform(partial(preprocess, processor=processor, label2id=label2id))

    # Without --pooling_mode and --pooler_add_layer_norm, keep the values of the checkpoint: the Lightning defaults
    # in a converted checkpoint, and the values of the run in a fine-tuned model.
    head_settings = {
        key: value
        for key, value in (
            ("pooling_mode", model_args.pooling_mode),
            ("pooler_add_layer_norm", model_args.pooler_add_layer_norm),
        )
        if value is not None
    }

    # Set the seed before the model is loaded, because the seed also sets the init of the classifier head.
    set_seed(training_args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
        head_init=model_args.head_init,
        reference_compile=model_args.reference_compile,
        **head_settings,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        processing_class=processor,  # saved with each checkpoint, so that a fine-tuned model can be evaluated
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_args.early_stopping_patience)],
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is None:
            load_best_checkpoint(trainer)  # the Trainer did not load it (save_strategy="best")
        trainer.save_model()  # the best checkpoint
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # The same as TextClassificationCLI.after_fit: validate and test the best checkpoint. Training is done, so
    # remove early stopping.
    trainer.remove_callback(EarlyStoppingCallback)
    if training_args.do_eval:
        metrics = trainer.evaluate(raw_datasets["validation"], metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        # Not trainer.evaluate: with persistent workers, it uses the kept validation loader again for each dataset,
        # so the test split gives the validation results. trainer.predict makes a new loader.
        metrics = trainer.predict(raw_datasets["test"], metric_key_prefix="test").metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
