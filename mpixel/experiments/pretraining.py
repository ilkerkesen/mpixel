from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import os.path as osp

from transformers import get_cosine_with_min_lr_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from mpixel.models import auto_register
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.utils import Modality, EXAMPLE_TEXTS

from mpixel.utils.visualization import generate_reconstruction_visualizations
from mpixel.datamodules.pretraining import FineWebDataModuleV1, FineWebIterableDataModule


class PixelPretrainingExperiment(L.LightningModule):
    """Masked-autoencoding pretraining experiment for MPIXEL."""

    def __init__(
        self,
        config_path: str,
        compile_model: bool = True,
        patch_size: int = 16,
        dropout: Optional[float] = None,
        mask_ratio: float = 0.25,
        norm_pix_loss: bool = True,
        use_lr_scheduler: bool = True,
        learning_rate: float = 3e-4,
        min_lr: float = 1e-5,
        adam_epsilon: float = 1e-8,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        warmup_steps: int = 25_000,
        weight_decay: float = 0.0,
        temperature: float = 1.0,
        max_num_patches: int = 512,
        patch_embedding_init: Optional[str] = None,
    ):
        # Initialize as LightningModule (skip parent's __init__ to override model)
        super().__init__()
        self.config_path = config_path
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.compile_model = compile_model
        self.use_lr_scheduler = use_lr_scheduler
        self.model_config = MPIXELConfig.from_pretrained(config_path)
        # from_pretrained strips attn_implementation from the JSON dict; re-apply it
        raw_cfg = json.loads(Path(config_path).read_text())
        if attn_impl := raw_cfg.get('attn_implementation'):
            self.model_config._attn_implementation = attn_impl
        self.model_config.patch_size = patch_size
        self.model_config.max_num_patches = max_num_patches
        self.model_config.mask_ratio = mask_ratio
        self.model_config.norm_pix_loss = norm_pix_loss
        self.model_config.temperature = temperature
        if patch_embedding_init is not None:
            self.model_config.patch_embedding_init = patch_embedding_init
        self.set_dropout(dropout=dropout)

        # Get dtype from config (can be string like "bfloat16" or torch.dtype)
        self.torch_dtype = getattr(self.model_config, 'dtype', None)
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = getattr(torch, self.torch_dtype, None)

        self.init_model()
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
        self.model_config.reference_compile = self.compile_model
        model = MPIXELForPreTraining._from_config(self.model_config)
        model.vit.embeddings.initialize_weights()
        if self.torch_dtype is not None:
            model = model.to(self.torch_dtype)
        self.model = model

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        model = self.model
        forbidden_name_patterns = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
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
            foreach=True,
        )

        if not self.use_lr_scheduler:
            return [optimizer]

        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.5,
            min_lr=self.hparams.min_lr,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, batch):
        kwargs = dict(
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            patch_mask=batch.get('patch_mask', None),
            return_dict=True,
        )
        output = self.model(**kwargs)
        return output

    def on_before_optimizer_step(self, optimizer):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log('grad_norm', total_norm, on_step=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.forward(batch)
        loss = output.loss
        self.log('trn_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        output = self.forward(batch)
        loss = output.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_config"] = self.model_config.to_dict()

    def on_fit_start(self):
        if self.trainer.global_rank != 0:
            return
        config_dir = self._resolve_checkpoint_dir()
        if config_dir is None:
            return
        config_path = config_dir / "model_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.model_config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _resolve_checkpoint_dir(self):
        checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
        dirpath = getattr(checkpoint_callback, "dirpath", None)
        if dirpath:
            return Path(osp.split(dirpath)[0])
        default_root_dir = getattr(self.trainer, "default_root_dir", None)
        if default_root_dir:
            return Path(default_root_dir)
        return None

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Generate and log reconstruction visualizations using EXAMPLE_TEXTS."""
        if self.trainer.global_rank != 0:
            return

        dm = self.trainer.datamodule
        if dm is None:
            return

        dataset = getattr(dm, "val_dataset", None) or getattr(dm, "train_dataset", None)
        if dataset is None:
            return

        self.model.eval()

        visualizations = generate_reconstruction_visualizations(
            model=self.model,
            texts=EXAMPLE_TEXTS,
            pixel_processor=dataset.pixel_processor,
            transforms=dataset.transforms,
            mask_generator=dataset.mask_generator,
            patch_size=dataset.patch_size,
            max_num_patches=dataset.max_num_patches,
            span_mask_ratio=dataset.span_mask_ratio,
            font_size=dataset.font_size,
            device=self.device,
        )

        for idx, img in visualizations:
            self.logger.experiment.add_image(
                f'reconstruction/{idx}',
                img,
                self.global_step,
                dataformats='HWC'
            )


class PretrainingCLI(LightningCLI):
    def before_instantiate_classes(self):
        config_path = self.config.fit.model.config_path
        model_cfg = json.loads(Path(config_path).read_text())

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(LearningRateMonitor, 'lr_monitor')

        parser.set_defaults({
            # Lightning checkpoint - save every N train steps
            'model_checkpoint.save_top_k': -1,
            'model_checkpoint.save_last': True,
            'model_checkpoint.every_n_train_steps': 10_000,

            # lr monitor
            'lr_monitor.logging_interval': 'step',
        })

        parser.link_arguments(
            "data.span_mask_ratio",
            "model.mask_ratio",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.patch_size",
            "model.patch_size",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.max_num_patches",
            "model.max_num_patches",
            apply_on="parse",
        )


if __name__ == "__main__":
    cli = PretrainingCLI(
        model_class=PixelPretrainingExperiment,
        datamodule_class=FineWebIterableDataModule,
        trainer_defaults={
            'accelerator': 'auto',
            'devices': 'auto',
            # 'strategy': 'ddp',
            'precision': 'bf16-mixed',
            'max_steps': 1_000_000,
            'num_sanity_val_steps': 2,
            'val_check_interval': 5000,
            'log_every_n_steps': 100,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'logger': {
                'class_path': 'lightning.pytorch.loggers.TensorBoardLogger',
                'init_args': {
                    'save_dir': '.',
                }
            },
        },
    )
