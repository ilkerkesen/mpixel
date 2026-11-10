import json
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from mpixel.models import auto_register
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.models.tasks.udp import PixelForBiaffineParsing, load_for_biaffine_parsing
from mpixel.datamodules.udp import UD_HEAD_LABELS, UDDataModule


def _compute_parsing_metrics(arc_logits, rel_logits, arc_labels, rel_labels, pad_token_id=-100):
    """
    Returns (arc_ok, las_ok, n_valid) as Python ints.

    Relation scores are read at the gold head index, matching the
    pixel-m4 PIXELTrainerForBiaffineParsing evaluation so val/test LAS
    stays comparable to prior numbers.

    arc_logits: (B, n_words, n_heads)
    rel_logits: (B, n_words, n_heads, n_labels)
    arc_labels:  (B, n_words)  — gold head indices, -100 for padding
    rel_labels:  (B, n_words)  — gold relation class ids, -100 for padding
    """
    mask = arc_labels.ne(pad_token_id)
    arc_preds = arc_logits.argmax(-1)   # (B, n_words)

    B, n_words, n_heads, n_labels = rel_logits.shape
    # Clamp pad (-100) to a valid index; these positions are dropped by `mask` below.
    gold_heads = arc_labels.clamp(min=0)
    idx = gold_heads.unsqueeze(-1).unsqueeze(-1).expand(B, n_words, 1, n_labels)
    rel_preds = rel_logits.gather(2, idx).squeeze(2).argmax(-1)  # (B, n_words)

    arc_ok = ((arc_preds == arc_labels) & mask).sum().item()
    las_ok = ((arc_preds == arc_labels) & (rel_preds == rel_labels) & mask).sum().item()
    n_valid = mask.sum().item()
    return arc_ok, las_ok, n_valid


class UDPExperiment(L.LightningModule):
    def __init__(
        self,
        model_id_or_path: str,
        labels: list = UD_HEAD_LABELS,
        compile_model: bool = False,
        dropout: float = 0.1,
        max_num_patches: int = 256,
        pad_token_id: int = -100,
        use_lr_scheduler: bool = True,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model_id_or_path = model_id_or_path
        self.compile_model = compile_model
        self.use_lr_scheduler = use_lr_scheduler
        self.num_labels = len(labels)
        self.max_num_patches = max_num_patches
        self.pad_token_id = pad_token_id

        self.init_model(dropout)
        self._reset_stats('val')
        self._reset_stats('test')
        self.save_hyperparameters()

    def init_model(self, dropout: float = 0.1):
        pretraining_class = MPIXELForPreTraining
        model = load_for_biaffine_parsing(
            checkpoint_path=self.model_id_or_path,
            pretraining_class=pretraining_class,
            num_labels=self.num_labels,
            dropout=dropout,
            pad_token_id=self.pad_token_id,
        )
        # NOTE: Not sure whether model compilation works.
        self.model = torch.compile(model) if self.compile_model else model

    def configure_optimizers(self):
        model = self.model
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )

        if not self.use_lr_scheduler:
            return [optimizer]

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, batch):
        return self.model(
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            word_starts=batch['word_starts'],
            arc_labels=batch.get('arc_labels'),
            rel_labels=batch.get('rel_labels'),
        )

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.forward(batch)
        self.log('trn_loss', output.loss, prog_bar=True)
        return output.loss

    @torch.no_grad()
    def inference_step_with_labels(self, batch, batch_idx, split: str = 'val'):
        self.model.eval()
        output = self.forward(batch)
        self.log(f'{split}_loss', output.loss, on_epoch=True, prog_bar=True)

        arc_ok, las_ok, n = _compute_parsing_metrics(
            output.arc_logits, output.rel_logits,
            batch['arc_labels'], batch['rel_labels'],
            pad_token_id=self.pad_token_id,
        )
        self._update_stats(split, arc_ok, las_ok, n)

    def validation_step(self, batch, batch_idx):
        return self.inference_step_with_labels(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.inference_step_with_labels(batch, batch_idx, split='test')

    def on_save_checkpoint(self, checkpoint):
        """Store all that a conversion to an HF model needs: the model config with the label names, and the
        rendering settings of the datamodule."""
        labels = self.hparams.labels
        model_config = deepcopy(self.model.config)
        model_config.id2label = dict(enumerate(labels))
        model_config.label2id = {label: i for i, label in enumerate(labels)}
        checkpoint["model_config"] = model_config.to_dict()
        datamodule = self.trainer.datamodule
        checkpoint["processor_config"] = {
            key: getattr(datamodule, key) for key in ("patch_size", "font_size", "max_num_patches", "renderer_type")
        }

    def on_validation_epoch_end(self):
        uas, las = self._get_metrics('val')
        self.log('val_uas', uas, prog_bar=True)
        self.log('val_las', las, prog_bar=True)
        self._reset_stats('val')

    def on_test_epoch_end(self):
        uas, las = self._get_metrics('test')
        self.log('test_uas', uas)
        self.log('test_las', las)
        self._reset_stats('test')

    def _reset_stats(self, split: str):
        setattr(self, f'_{split}_arc_ok', 0)
        setattr(self, f'_{split}_las_ok', 0)
        setattr(self, f'_{split}_total', 0)

    def _update_stats(self, split: str, arc_ok: int, las_ok: int, total: int):
        setattr(self, f'_{split}_arc_ok', getattr(self, f'_{split}_arc_ok') + arc_ok)
        setattr(self, f'_{split}_las_ok', getattr(self, f'_{split}_las_ok') + las_ok)
        setattr(self, f'_{split}_total', getattr(self, f'_{split}_total') + total)

    def _get_metrics(self, split: str):
        total = getattr(self, f'_{split}_total')
        if total == 0:
            return 0.0, 0.0
        uas = getattr(self, f'_{split}_arc_ok') / total
        las = getattr(self, f'_{split}_las_ok') / total
        return uas, las


class UDPCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')

        parser.set_defaults({
            'model_checkpoint.monitor': 'val_las',
            'model_checkpoint.mode': 'max',
            'model_checkpoint.save_top_k': 2,

            'early_stopping.monitor': 'val_las',
            'early_stopping.mode': 'max',
            'early_stopping.patience': 5,

            'data.batch_size': 64,
        })

        parser.link_arguments(
            "data.labels",
            "model.labels",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.max_num_patches",
            "model.max_num_patches",
            apply_on="instantiate",
        )

    def after_fit(self):
        val_results = self.trainer.validate(ckpt_path="best", datamodule=self.datamodule)
        test_results = self.trainer.test(ckpt_path="best", datamodule=self.datamodule)

        results = {}
        if val_results:
            results.update({k: float(v) for k, v in val_results[0].items()})
        if test_results:
            results.update({k: float(v) for k, v in test_results[0].items()})

        log_dir = self.trainer.log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            output_path = os.path.join(log_dir, "all_results.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    cli = UDPCLI(
        model_class=UDPExperiment,
        datamodule_class=UDDataModule,
        trainer_defaults={
            'accelerator': 'auto',
            'devices': [0],
            'precision': 'bf16-mixed',
            'max_steps': 15_000,
            'num_sanity_val_steps': 2,
            'val_check_interval': 500,
            'check_val_every_n_epoch': None,
            'gradient_clip_val': 1.0,
            'logger': {
                'class_path': 'lightning.pytorch.loggers.TensorBoardLogger',
                'init_args': {
                    'save_dir': './lightning_logs/udp',
                }
            }
        },
    )
