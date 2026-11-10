from collections import defaultdict

import torch
import torch.nn.functional as F

from mpixel.experiments.text_classification import (
    TextClassificationExperiment,
    TextClassificationCLI,
)
from mpixel.datamodules.xnli import XNLIDataModule


class XNLIExperiment(TextClassificationExperiment):
    """XNLI translate-train-all: each batch carries a per-example ``language``
    tag, so per-language accuracy is bucketed in a single eval pass and reported
    (per language, pooled, and averaged) at epoch end."""

    def _reset_lang_counts(self):
        self._lang_correct = defaultdict(int)
        self._lang_total = defaultdict(int)

    def on_validation_epoch_start(self):
        self._reset_lang_counts()

    def on_test_epoch_start(self):
        self._reset_lang_counts()

    @torch.no_grad()
    def _eval_step(self, batch, split_name):
        self.model.eval()
        logits = self.forward(batch).logits
        labels = batch['labels']
        loss = F.cross_entropy(logits, labels)
        self.log(f'{split_name}_loss', loss, on_epoch=True, prog_bar=True)
        correct = (logits.argmax(dim=-1) == labels).tolist()
        for language, is_correct in zip(batch['language'], correct):
            self._lang_correct[language] += int(is_correct)
            self._lang_total[language] += 1

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, 'test')

    def _log_lang_acc(self, split_name):
        accs = {
            lang: self._lang_correct[lang] / self._lang_total[lang]
            for lang in self._lang_total
        }
        for language, acc in accs.items():
            self.log(f'{split_name}_acc/{language}', acc)
        # pooled (micro) accuracy drives checkpoint selection; also report the
        # per-language average (the canonical translate-train-all number).
        pooled = sum(self._lang_correct.values()) / sum(self._lang_total.values())
        self.log(f'{split_name}_acc', pooled, prog_bar=True)
        self.log(f'{split_name}_acc_avg', sum(accs.values()) / len(accs))

    def on_validation_epoch_end(self):
        self._log_lang_acc('val')

    def on_test_epoch_end(self):
        self._log_lang_acc('test')


class XNLICLI(TextClassificationCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        # XNLI's canonical metric is accuracy; select/early-stop on val_acc.
        parser.set_defaults({
            'model_checkpoint.monitor': 'val_acc',
            'early_stopping.monitor': 'val_acc',
        })


if __name__ == "__main__":
    cli = XNLICLI(
        model_class=XNLIExperiment,
        datamodule_class=XNLIDataModule,
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
                    'save_dir': './lightning_logs/xnli',
                }
            }
        },
    )
