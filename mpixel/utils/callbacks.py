import shutil
from typing import Optional
from pathlib import Path

import lightning as L


class HuggingFaceCheckpoint(L.pytorch.callbacks.Callback):
    """Saves HuggingFace-compatible checkpoints during training.

    Supports two modes (can be combined):
    - Best model saving: monitor a metric and keep the top_k best checkpoints.
    - Periodic saving: save every ``every_n_train_steps`` training steps.
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        monitor: Optional[str] = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        every_n_train_steps: int = 0,
    ):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.every_n_train_steps = every_n_train_steps
        self._best: list[tuple[float, Path]] = []

    def _save_dir(self, trainer) -> Path:
        if self.dirpath:
            return Path(self.dirpath)
        return Path(trainer.log_dir) / "huggingface"

    def _save(self, trainer, pl_module, name: str) -> Path:
        path = self._save_dir(trainer) / name
        model = pl_module.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.save_pretrained(path)
        pl_module.model_config.save_pretrained(path)
        return path

    def _prune(self):
        if self.save_top_k < 0:
            return
        reverse = self.mode == "max"
        self._best.sort(key=lambda x: x[0], reverse=reverse)
        while len(self._best) > self.save_top_k:
            _, path = self._best.pop()
            if path.exists():
                shutil.rmtree(path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_train_steps <= 0:
            return
        if trainer.global_rank != 0:
            return
        if trainer.global_step % self.every_n_train_steps != 0:
            return
        self._save(trainer, pl_module, f"checkpoint-{trainer.global_step}")

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if trainer.sanity_checking:
            return
        if self.save_top_k == 0 or not self.monitor:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        score = current.item()
        path = self._save(trainer, pl_module, f"checkpoint-{trainer.global_step}")
        self._best.append((score, path))
        self._prune()
