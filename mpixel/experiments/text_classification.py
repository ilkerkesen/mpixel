import json
import os
import os.path as osp
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.classification import F1Score
from sklearn.metrics import f1_score

from mpixel.models import auto_register
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.models.pooling import PoolingMode
from mpixel.models.tasks.classification import (
    PixelForSequenceClassification,
    load_for_sequence_classification,
)
from mpixel.utils import Modality
from mpixel.utils.checkpoint import config_from_checkpoint, load_hf_model_config
from mpixel.datamodules.sib200 import SIB200DataModule


def _find_config(model_id_or_path: str) -> str:
    """If model_id_or_path is a Lightning .ckpt, look for model_config.json in
    the log directory (grandparent of checkpoints/). Otherwise pass through."""
    if model_id_or_path.endswith(".ckpt"):
        log_dir = os.path.dirname(os.path.dirname(model_id_or_path))
        if os.path.isfile(os.path.join(log_dir, "model_config.json")):
            return log_dir
    return model_id_or_path


def _load_model_config(
    model_id_or_path: str,
    model_config_path: Optional[str],
    num_labels: int,
):
    if model_config_path is not None:
        return AutoConfig.from_pretrained(model_config_path, num_labels=num_labels)

    if model_id_or_path.endswith(".ckpt"):
        checkpoint = torch.load(model_id_or_path, map_location="cpu", weights_only=False)
        if "model_config" in checkpoint:
            config = config_from_checkpoint(checkpoint["model_config"])
            config.num_labels = num_labels
            return config
        return AutoConfig.from_pretrained(_find_config(model_id_or_path), num_labels=num_labels)

    return load_hf_model_config(model_id_or_path, num_labels=num_labels)


class TextClassificationExperiment(L.LightningModule):
    def __init__(
        self,
        model_id_or_path: str,
        num_labels: int,
        model_config_path: Optional[str] = None,
        compile_model: bool = False,
        dropout: Optional[float] = None,
        max_num_patches: int = 256,
        pooling_mode: str = 'mean',
        pooler_add_layer_norm: bool = True,
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
        self.model_config_path = model_config_path
        self.compile_model = compile_model
        self.use_lr_scheduler = use_lr_scheduler
        self.num_labels = num_labels
        self.max_num_patches = max_num_patches
        self.pooling_mode = PoolingMode.from_string(pooling_mode)
        self.pooler_add_layer_norm = pooler_add_layer_norm

        # Load config and apply dropout. Lightning pretraining checkpoints
        # carry the authoritative config in checkpoint["model_config"]; reading
        # the log directory via AutoConfig.from_pretrained can pick up the
        # wrong class when only model_config.json is present.
        self.model_config = _load_model_config(model_id_or_path, model_config_path, num_labels)
        self.set_dropout(dropout)
        self.init_model()

        self.valid_f1 = F1Score(task='multiclass', num_classes=self.num_labels, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=self.num_labels, average='macro')
        self.save_hyperparameters()

    @property
    def modality(self):
        return Modality.IMAGE

    def set_dropout(self, dropout: Optional[float]):
        if dropout is None:
            return
        self.model_config.attention_probs_dropout_prob = dropout
        self.model_config.hidden_dropout_prob = dropout

    def init_model(self):
        pretraining_class = MPIXELForPreTraining
        model = load_for_sequence_classification(
            checkpoint_path=self.model_id_or_path,
            config=self.model_config,
            pretraining_class=pretraining_class,
            num_labels=self.num_labels,
            pooling_mode=self.pooling_mode,
            add_layer_norm=self.pooler_add_layer_norm,
        )
        # NOTE: not sure whether model compilation really works.
        self.model = torch.compile(model) if self.compile_model else model

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and linear decay)."""
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
        output = self.model(
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )
        return output

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.forward(batch)
        loss = F.cross_entropy(output.logits, batch['labels'])
        acc = multiclass_accuracy(
            preds=output.logits,
            target=batch['labels'],
            num_classes=self.num_labels,
            average='micro',
        )
        self.log('trn_loss', loss, prog_bar=True)
        self.log('trn_acc', acc, prog_bar=True)
        return loss

    @torch.no_grad()
    def inference_step_with_labels(self, batch, batch_idx, split_name='val'):
        self.model.eval()
        output = self.forward(batch)
        loss = F.cross_entropy(output.logits, batch['labels'])

        acc = multiclass_accuracy(
            preds=output.logits,
            target=batch['labels'],
            num_classes=self.num_labels,
            average='micro',
        )

        self.log(f'{split_name}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{split_name}_acc', acc, on_epoch=True, prog_bar=True)
        if split_name == 'val':
            self.valid_f1.update(output.logits, batch['labels'])
        elif split_name == 'test':
            self.test_f1.update(output.logits, batch['labels'])

    def validation_step(self, batch, batch_idx):
        return self.inference_step_with_labels(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.inference_step_with_labels(batch, batch_idx, split_name='test')

    def on_save_checkpoint(self, checkpoint):
        """Store all that a conversion to an HF model needs: the model config with the head settings and the label
        names, and the rendering settings of the datamodule."""
        datamodule = self.trainer.datamodule
        model_config = deepcopy(self.model_config)
        model_config.pooling_mode = self.hparams.pooling_mode  # the string, which keeps the PMA head count
        model_config.pooler_add_layer_norm = self.pooler_add_layer_norm
        model_config.id2label = dict(datamodule.id2label)
        model_config.label2id = dict(datamodule.label2id)
        checkpoint["model_config"] = model_config.to_dict()
        checkpoint["processor_config"] = {
            key: getattr(datamodule, key) for key in ("patch_size", "font_size", "max_num_patches", "renderer_type")
        }

    def on_validation_epoch_end(self):
        self.log('val_f1', self.valid_f1.compute(), prog_bar=True)
        self.valid_f1.reset()

    def on_test_epoch_end(self):
        self.log('test_f1', self.test_f1.compute())
        self.test_f1.reset()


class TextClassificationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')

        parser.set_defaults({
            # model checkpoint args
            'model_checkpoint.monitor': 'val_f1',
            'model_checkpoint.mode': 'max',
            'model_checkpoint.save_top_k': 2,

            # early stopping configuration
            'early_stopping.monitor': 'val_f1',
            'early_stopping.mode': 'max',
            'early_stopping.patience': 20,
        })

        parser.link_arguments(
            "data.num_labels",
            "model.num_labels",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.max_num_patches",
            "model.max_num_patches",
            apply_on="parse",
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
    cli = TextClassificationCLI(
        model_class=TextClassificationExperiment,
        datamodule_class=SIB200DataModule,
        trainer_defaults={
            'accelerator': 'auto',
            'devices': [0],
            'precision': 'bf16-mixed',
            'max_steps': 15_000,
            'num_sanity_val_steps': 2,
            'val_check_interval': 1.0,
            'check_val_every_n_epoch': 1,
            'gradient_clip_val': 1.0,
            'logger': {
                'class_path': 'lightning.pytorch.loggers.TensorBoardLogger',
                'init_args': {
                    'save_dir': './lightning_logs/sib-200',
                }
            }
        },
    )
