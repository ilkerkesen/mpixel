from typing import Any, Dict, List, Optional
import os.path as osp

import numpy as np
import torch
import lightning as L
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from font_download import FontConfig
from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED
from pixel_renderer import PixelRendererProcessor

from mpixel.utils.preprocessing import get_attention_mask, get_transforms
from mpixel.renderer.functional import _render_units
from mpixel.renderer.processor import _text_to_units


# NER datasets are pre-tokenized; we render each token independently and align
# its label to the patch where the token starts. All three sources prepared by
# scripts/downstream (naamapadam-*, klue-ner, universal-ner-*) share the same
# on-disk format: a DatasetDict of {tokens, ner_tags} with ner_tags as a
# ClassLabel, so a single datamodule covers them, parameterized by `dataset`.

_STRATEGY_BY_RENDERER = {
    "bigrams": "bigrams",
    "adaptive": "auto",
    "words": "words",
}


def _token_to_units(token: str, renderer_type: str) -> List[str]:
    """Chunk a single token into rendering units in logical (input) order.

    Whitespace-only tokens (e.g. KLUE's literal space tokens) produce no units
    and are therefore not rendered and receive no label patch.
    """
    strategy = _STRATEGY_BY_RENDERER.get(renderer_type)
    if strategy is None:
        raise ValueError(f"Unknown renderer_type: {renderer_type!r}")
    units, _ = _text_to_units(token, strategy)
    return units


class NERDataset(Dataset):
    def __init__(
        self,
        data,
        patch_size: int,
        max_num_patches: int,
        font_size: int,
        renderer_type: str = "adaptive",
        pad_label_id: int = -100,
    ) -> None:
        super().__init__()
        self.data = data
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.font_size = font_size
        self.renderer_type = renderer_type
        self.pad_label_id = pad_label_id
        self.font_config = FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)
        self._font_owner = PixelRendererProcessor(font=self.font_config)
        self.transforms = get_transforms(do_resize=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        try:
            return self._process(item["tokens"], item["ner_tags"])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    def _process(self, tokens: List[str], ner_tags: List[int]) -> Dict[str, Any]:
        self._font_owner._ensure_fontconfig_initialized()

        # Render tokens in input order; whitespace tokens drop out here.
        units: List[str] = []
        unit_counts: List[int] = []
        kept_tokens: List[str] = []
        kept_tags: List[int] = []
        for token, tag in zip(tokens, ner_tags):
            token_units = _token_to_units(token, self.renderer_type)
            if not token_units:
                continue
            units.extend(token_units)
            unit_counts.append(len(token_units))
            kept_tokens.append(token)
            kept_tags.append(tag)

        image_np, patch_counts = _render_units(
            units,
            block_size=self.patch_size,
            font_size=self.font_size,
            pad=1,
        )
        # Append a black SEP patch (matches the high-level processors' contract).
        sep_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        image_np = np.concatenate([image_np, sep_patch], axis=1)

        # word_starts_all[i] = first patch index of kept token i (pre-truncation).
        word_starts_all: List[int] = []
        cursor = 0
        offset = 0
        for count in unit_counts:
            word_starts_all.append(cursor)
            cursor += sum(patch_counts[offset:offset + count])
            offset += count
        num_text_patches_rendered = cursor

        max_text_patches = self.max_num_patches - 1   # reserve last slot for SEP
        num_text_patches = min(num_text_patches_rendered, max_text_patches)

        # Drop trailing tokens whose patch span doesn't fully fit before SEP.
        boundaries = word_starts_all + [num_text_patches_rendered]
        n_words = 0
        while n_words < len(word_starts_all) and boundaries[n_words + 1] <= num_text_patches:
            n_words += 1
        word_starts = word_starts_all[:n_words]
        num_text_patches = boundaries[n_words] if n_words > 0 else 0

        # Build the padded strip (white bg + text + SEP + padding).
        img_tensor = self.transforms(Image.fromarray(image_np))   # (C, H, W_with_sep)
        c, h = img_tensor.shape[0], img_tensor.shape[1]
        max_width = self.max_num_patches * self.patch_size
        pixel_values = torch.ones((c, h, max_width))
        text_width = num_text_patches * self.patch_size
        pixel_values[:, :, :text_width] = img_tensor[:, :, :text_width]
        pixel_values[:, :, text_width:text_width + self.patch_size] = 0.0

        attention_mask = get_attention_mask(num_text_patches, seq_length=self.max_num_patches)

        # Per-patch labels: the token's label id at its start patch, -100 elsewhere
        # (matches the old run_ner_bigrams.py contract).
        labels = [self.pad_label_id] * self.max_num_patches
        for word_start, tag in zip(word_starts, kept_tags):
            labels[word_start] = tag

        return {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_text_patches": num_text_patches,
            "tokens": kept_tokens[:n_words],
        }


class NERDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        root_data_dir: str = "./data/ner",
        batch_size: int = 32,
        num_workers: int = 8,
        max_num_patches: int = 256,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
        pad_label_id: int = -100,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.root_data_dir = osp.abspath(osp.expanduser(root_data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type
        self.pad_label_id = pad_label_id

        # Expose labels at construction time (read off disk; mmap'd, cheap) so the
        # CLI can link data.labels -> model.labels before setup() runs.
        features = load_from_disk(self.dataset_dir)["train"].features
        self.labels = features["ner_tags"].feature.names
        self.num_labels = len(self.labels)
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

    @property
    def dataset_dir(self) -> str:
        return osp.join(self.root_data_dir, self.dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = load_from_disk(self.dataset_dir)
        self.train_dataset = self.setup_split(self.data["train"])
        self.val_dataset = self.setup_split(self.data["validation"])
        self.test_dataset = self.setup_split(self.data["test"]) if "test" in self.data else None

    def setup_split(self, dataset) -> NERDataset:
        dataset = dataset.filter(lambda x: len(x["tokens"]) > 0)
        return NERDataset(
            data=dataset,
            patch_size=self.patch_size,
            max_num_patches=self.max_num_patches,
            font_size=self.font_size,
            renderer_type=self.renderer_type,
            pad_label_id=self.pad_label_id,
        )

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])
        num_text_patches = torch.tensor(
            [example["num_text_patches"] for example in examples],
            dtype=torch.long,
        )
        batch = {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_text_patches": num_text_patches,
            "tokens": [example["tokens"] for example in examples],
        }
        return batch

    def base_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self.base_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.base_dataloader(self.val_dataset)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self.base_dataloader(self.test_dataset)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "universal-ner-english-ewt"
    data_module = NERDataModule(dataset=dataset, num_workers=0)
    data_module.setup()
    print(f"dataset={dataset} num_labels={data_module.num_labels} labels={data_module.labels}")
    batch = data_module.collate_fn([data_module.train_dataset[i] for i in range(4)])
    for key, value in batch.items():
        shape = tuple(value.shape) if torch.is_tensor(value) else f"list[{len(value)}]"
        print(f"  {key}: {shape}")
