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
from torchmetrics.classification import (
    BinaryF1Score,
    MatthewsCorrCoef,
    MulticlassAccuracy,
)
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

from mpixel.models import auto_register
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.models.pooling import PoolingMode
from mpixel.models.tasks.classification import (
    PixelForSequenceClassification,
    load_for_sequence_classification,
)
from mpixel.utils import Modality
from mpixel.datamodules.glue import GLUEDataModule, TASK_NUM_LABELS
from mpixel.experiments.text_classification import _load_model_config


class GLUEExperiment(L.LightningModule):
    def __init__(
        self,
        model_id_or_path: str,
        num_labels: int,
        task_name: str = "sst2",
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
        self.task_name = task_name
        self.is_regression = task_name == "stsb"
        self.max_num_patches = max_num_patches
        self.pooling_mode = PoolingMode.from_string(pooling_mode)
        self.pooler_add_layer_norm = pooler_add_layer_norm

        self.model_config = _load_model_config(model_id_or_path, model_config_path, num_labels)
        self.set_dropout(dropout)
        self.init_model()
        self._init_metrics()
        self.save_hyperparameters()

    def _init_metrics(self):
        if self.is_regression:
            self.val_metric = SpearmanCorrCoef()
            self.test_metric = SpearmanCorrCoef()
            self.val_pearson = PearsonCorrCoef()
            self.test_pearson = PearsonCorrCoef()
        elif self.task_name == "cola":
            self.val_metric = MatthewsCorrCoef(task="multiclass", num_classes=self.num_labels)
            self.test_metric = MatthewsCorrCoef(task="multiclass", num_classes=self.num_labels)
        elif self.task_name in ("mrpc", "qqp"):
            self.val_metric = BinaryF1Score()
            self.test_metric = BinaryF1Score()
        else:
            self.val_metric = MulticlassAccuracy(num_classes=self.num_labels, average="micro")
            self.test_metric = MulticlassAccuracy(num_classes=self.num_labels, average="micro")

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
        # NOTE: Not sure whether model compliation works.
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
        output = self.model(
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )
        return output

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.forward(batch)
        if self.is_regression:
            loss = F.mse_loss(output.logits.squeeze(-1), batch['labels'])
        else:
            loss = F.cross_entropy(output.logits, batch['labels'])
        self.log('trn_loss', loss, prog_bar=True)
        if not self.is_regression:
            acc = multiclass_accuracy(
                preds=output.logits,
                target=batch['labels'],
                num_classes=self.num_labels,
                average='micro',
            )
            self.log('trn_acc', acc, prog_bar=True)
        return loss

    @torch.no_grad()
    def inference_step_with_labels(self, batch, batch_idx, split_name='val'):
        self.model.eval()
        output = self.forward(batch)
        if self.is_regression:
            loss = F.mse_loss(output.logits.squeeze(-1), batch['labels'])
            preds = output.logits.squeeze(-1)
        else:
            loss = F.cross_entropy(output.logits, batch['labels'])

        self.log(f'{split_name}_loss', loss, on_epoch=True, prog_bar=True)

        if not self.is_regression:
            acc = multiclass_accuracy(
                preds=output.logits,
                target=batch['labels'],
                num_classes=self.num_labels,
                average='micro',
            )
            self.log(f'{split_name}_acc', acc, on_epoch=True, prog_bar=True)

        if self.task_name in ("mrpc", "qqp"):
            metric_preds = output.logits.argmax(-1)
        elif self.is_regression:
            metric_preds = preds
        else:
            metric_preds = output.logits

        if split_name == 'val':
            self.val_metric.update(metric_preds, batch['labels'])
            if self.is_regression:
                self.val_pearson.update(metric_preds, batch['labels'])
        elif split_name == 'test':
            self.test_metric.update(metric_preds, batch['labels'])
            if self.is_regression:
                self.test_pearson.update(metric_preds, batch['labels'])

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
        label_feature = datamodule.data["train"].features["label"]
        if hasattr(label_feature, "names"):  # STS-B is a regression task and has no label names
            model_config.id2label = dict(enumerate(label_feature.names))
            model_config.label2id = {label: i for i, label in enumerate(label_feature.names)}
        checkpoint["model_config"] = model_config.to_dict()
        checkpoint["processor_config"] = {
            key: getattr(datamodule, key) for key in ("patch_size", "font_size", "max_num_patches", "renderer_type")
        }

    def on_validation_epoch_end(self):
        self.log('val_metric', self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()
        if self.is_regression:
            self.log('val_pearson', self.val_pearson.compute(), prog_bar=True)
            self.val_pearson.reset()

    def on_test_epoch_end(self):
        self.log('test_metric', self.test_metric.compute())
        self.test_metric.reset()
        if self.is_regression:
            self.log('test_pearson', self.test_pearson.compute())
            self.test_pearson.reset()


class GLUECLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')

        parser.set_defaults({
            'model_checkpoint.monitor': 'val_metric',
            'model_checkpoint.mode': 'max',
            'model_checkpoint.save_top_k': 2,

            'early_stopping.monitor': 'val_metric',
            'early_stopping.mode': 'max',
            'early_stopping.patience': 5,
        })

        parser.link_arguments(
            "data.task_name",
            "model.task_name",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.num_labels",
            "model.num_labels",
            apply_on="instantiate",
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

        # MNLI mismatched evaluation
        if hasattr(self.datamodule, 'val_mm_dataset'):
            mm_loader = self.datamodule.base_dataloader(self.datamodule.val_mm_dataset)
            mm_results = self.trainer.test(ckpt_path="best", dataloaders=mm_loader)
            if mm_results:
                results.update({f"mm_{k}": float(v) for k, v in mm_results[0].items()})

        log_dir = self.trainer.log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            output_path = os.path.join(log_dir, "all_results.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    cli = GLUECLI(
        model_class=GLUEExperiment,
        datamodule_class=GLUEDataModule,
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
                    'save_dir': './lightning_logs/glue',
                }
            }
        },
    )
