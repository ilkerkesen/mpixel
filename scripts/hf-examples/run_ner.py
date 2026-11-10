#!/usr/bin/env python
"""
Fine-tune a converted MPIXEL checkpoint on one NER dataset with the HuggingFace Trainer.

This script is the Trainer version of the Lightning experiment `mpixel/experiments/ner.py`. Its defaults are the
defaults of that experiment, so that a run uses the same setup as the reported runs. The data comes from
`NERDataModule`, the same code as the Lightning runs: it renders each token, and puts the label of the token on
the first patch of the token. All other patches get the label -100, which the loss and the metrics ignore.

Prepare the dataset with a script in `scripts/downstream/`, for example:

    python scripts/downstream/download_klue_ner.py --output-dir ./data/ner

Run it on one GPU, the same as the Lightning runs:

    CUDA_VISIBLE_DEVICES=0 python scripts/hf-examples/run_ner.py \
        --model_name_or_path <converted-checkpoint-dir> \
        --dataset klue-ner \
        --output_dir runs/ner/klue-ner/seed0 \
        --seed 0

Make the checkpoint with `scripts/utils/convert_to_huggingface.py`. All `TrainingArguments` options are
available, for example `--learning_rate` or `--max_steps`. You can also give all options in one JSON file:

    python scripts/hf-examples/run_ner.py args.json

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
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import mpixel.models.auto_register  # noqa: F401
from mpixel.datamodules.ner import NERDataModule


logger = logging.getLogger(__name__)

PAD_LABEL_ID = -100  # NERDataModule pad_label_id, the label of the patches without a label

# Settings of the Lightning NER runs that are different from the TrainingArguments defaults. These settings are
# already the same in both: AdamW parameter groups, learning rate, betas, epsilon, weight decay, linear schedule
# without warmup, and gradient clipping at norm 1.0.
LIGHTNING_DEFAULTS = dict(
    optim="adamw_torch",  # torch.optim.AdamW, not the fused version
    bf16=True,  # precision "bf16-mixed"
    max_steps=15_000,
    per_device_train_batch_size=32,  # NERDataModule batch_size
    per_device_eval_batch_size=32,
    eval_strategy="steps",  # validation every 500 steps (val_check_interval=500, check_val_every_n_epoch=None)
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # ModelCheckpoint save_top_k=2
    load_best_model_at_end=True,  # validate and test the best checkpoint
    metric_for_best_model="f1",  # ModelCheckpoint monitor="val_f1", mode="max"
    greater_is_better=True,
    dataloader_num_workers=8,  # NERDataModule num_workers
    logging_steps=50,  # Trainer log_every_n_steps
    report_to="tensorboard",  # TensorBoardLogger
    seed=0,
    do_train=True,  # train, then validate and test the best checkpoint
    do_eval=True,
    do_predict=True,
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to a converted MPIXEL checkpoint (scripts/utils/convert_to_huggingface.py), or to a "
            "model that this script fine-tuned."
        }
    )
    dropout: float = field(default=0.1, metadata={"help": "Hidden and attention dropout."})


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "NER dataset folder in root_data_dir, for example klue-ner or naamapadam-hi."}
    )
    root_data_dir: str = field(
        default="./data/ner", metadata={"help": "The folder of the scripts/downstream datasets."}
    )
    max_num_patches: int = field(
        default=256,  # NERDataModule max_num_patches
        metadata={"help": "Maximum number of patches of a rendered sentence."},
    )


@dataclass
class EarlyStoppingArguments:
    early_stopping_patience: int = field(
        default=5, metadata={"help": "Stop when the validation F1 does not improve for this many evaluations."}
    )


def parse_args(args=None):
    """Parse the command line, or one JSON file with the same options. Both use the Lightning defaults."""
    parser = HfArgumentParser((ModelArguments, DataArguments, EarlyStoppingArguments, TrainingArguments))
    parser.set_defaults(**LIGHTNING_DEFAULTS)
    args = sys.argv[1:] if args is None else args
    if len(args) == 1 and args[0].endswith(".json"):
        # parse_json_file does not use the parser defaults, so give them to parse_dict with the JSON values.
        with open(args[0]) as f:
            return parser.parse_dict({**LIGHTNING_DEFAULTS, **json.load(f)})
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


def get_datamodule(data_args: DataArguments) -> NERDataModule:
    """The NERDataModule of the Lightning runs, with its train, validation and (if the dataset has one) test split."""
    datamodule = NERDataModule(
        dataset=data_args.dataset,
        root_data_dir=data_args.root_data_dir,
        max_num_patches=data_args.max_num_patches,
    )
    datamodule.setup()
    return datamodule


def collate_fn(examples):
    """The model inputs of NERDataModule.collate_fn."""
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "attention_mask": torch.stack([example["attention_mask"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
    }


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The predicted label of each patch, the same as NERExperiment. The Trainer then keeps ids, not logits."""
    return logits.argmax(-1)


def compute_metrics(eval_pred: EvalPrediction, id2label: dict):
    """The seqeval metrics of NERExperiment, on the patches that have a label."""
    golds, preds = [], []
    for gold_ids, pred_ids in zip(eval_pred.label_ids, eval_pred.predictions):
        keep = gold_ids != PAD_LABEL_ID
        golds.append([id2label[i] for i in gold_ids[keep].tolist()])
        preds.append([id2label[i] for i in pred_ids[keep].tolist()])
    return {
        "accuracy": accuracy_score(golds, preds),
        "precision": precision_score(golds, preds),
        "recall": recall_score(golds, preds),
        "f1": f1_score(golds, preds),
    }


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
            "The Lightning NER runs used one GPU. More GPUs change the batch size. "
            "Run on one GPU, for example with CUDA_VISIBLE_DEVICES=0."
        )
    resume_checkpoint = get_resume_checkpoint(training_args) if training_args.do_train else None

    # The label ids are the ClassLabel ids of the dataset, the same as in the Lightning runs.
    datamodule = get_datamodule(data_args)

    # Set the seed before the model is loaded, because the seed also sets the init of the classifier head.
    set_seed(training_args.seed)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=datamodule.num_labels,
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, id2label=datamodule.id2label),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_args.early_stopping_patience)],
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model()  # the best checkpoint, because load_best_model_at_end=True
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # The same as NERCLI.after_fit: validate and test the best checkpoint. Some datasets (for example klue-ner)
    # have no test split. Training is done, so remove early stopping; on the test split it does not find eval_f1
    # and logs a warning.
    trainer.remove_callback(EarlyStoppingCallback)
    splits = []
    if training_args.do_eval:
        splits.append(("eval", datamodule.val_dataset))
    if training_args.do_predict:
        if datamodule.test_dataset is None:
            logger.info(f"{data_args.dataset} has no test split, so the script does not test.")
        else:
            splits.append(("test", datamodule.test_dataset))
    for split, dataset in splits:
        metrics = trainer.evaluate(dataset, metric_key_prefix=split)
        trainer.log_metrics(split, metrics)
        trainer.save_metrics(split, metrics)


if __name__ == "__main__":
    main()
