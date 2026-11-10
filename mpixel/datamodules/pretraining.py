from typing import Optional, Any, List, Dict
import os
import os.path as osp
import glob
from pathlib import Path
import math
import torch
import re
import random
from collections import defaultdict
from datasets import load_dataset, load_from_disk
import lightning as L
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image
from font_download import FontConfig
from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED
from pixel_renderer import PixelRendererProcessor
from mpixel.renderer.processor import BigramsRenderingProcessor, AdaptiveRenderingProcessor, WordsRenderingProcessor

from mpixel.models.masking import SpanMaskingGenerator
from mpixel.utils import get_fallback_fonts_dir
from mpixel.utils.preprocessing import get_attention_mask, get_transforms


def preprocess_text_for_pretraining(
    text: str,
    pixel_processor: PixelRendererProcessor | BigramsRenderingProcessor | AdaptiveRenderingProcessor | WordsRenderingProcessor,
    transforms,
    patch_size: int,
    max_num_patches: int,
    mask_generator: SpanMaskingGenerator,
    span_mask_ratio: float,
    font_size: int,
) -> Dict[str, Any]:
    """
    Shared preprocessing function for pretraining data.

    Renders text, applies transforms, pads to max width, creates attention mask,
    and generates span mask.

    Returns:
        Dictionary with keys: pixel_values, attention_mask, num_text_patches, patch_mask
    """
    # Render text
    if isinstance(pixel_processor, PixelRendererProcessor):
        rendered = pixel_processor.render_text_image(
            text=text,
            block_size=patch_size,
            font_size=font_size,
        )
        render_result = None
    else:
        render_result = pixel_processor(text=text, block_size=patch_size, font_size=font_size)
        rendered = Image.fromarray(render_result["pixel_values"])

    # Apply transforms
    img = transforms(rendered)
    c, h, w = img.shape

    # Pad to max width.
    # render_result is not None for BigramsRenderingProcessor / AdaptiveRenderingProcessor,
    # which append an explicit black SEP patch as the last patch of the rendered image.
    # We reserve the last slot for SEP so long text never pushes it out of the sequence.
    # PixelRendererProcessor (render_result is None) returns a plain PIL Image with no SEP;
    # it uses the legacy path where white padding implicitly fills the gap.
    max_width = max_num_patches * patch_size
    if render_result is not None:
        text_w = min(w - patch_size, (max_num_patches - 1) * patch_size)
        num_text_patches = text_w // patch_size
        pixel_values = torch.ones((c, h, max_width))
        pixel_values[:, :, :text_w] = img[:, :, :text_w]
        pixel_values[:, :, text_w:text_w + patch_size] = img[:, :, w - patch_size:w]
    else:
        actual_w = min(w, max_width)
        num_text_patches = actual_w // patch_size
        pixel_values = torch.ones((c, h, max_width))
        pixel_values[:, :, :actual_w] = img[:, :, :actual_w]

    # Create attention mask
    attention_mask = get_attention_mask(num_text_patches, seq_length=max_num_patches)

    # Generate span mask over text region (not full max_num_patches padding)
    effective_seq_length = num_text_patches + 1  # text + SEP
    num_patches_to_mask = math.ceil(effective_seq_length * span_mask_ratio)
    patch_mask = mask_generator(
        num_patches_total=effective_seq_length,  # Spans only in text+SEP region
        num_text_patches=num_text_patches + 1,   # +1 for SEP, consistent with legacy
        num_patches_to_mask=num_patches_to_mask
    )
    # Pad to max_num_patches (zeros for padding region)
    patch_mask = np.pad(patch_mask, (0, max_num_patches - effective_seq_length), constant_values=0)
    patch_mask = torch.tensor(patch_mask)

    out = {
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
        'num_text_patches': num_text_patches,
        'patch_mask': patch_mask,
    }

    return out


class FineWebDatasetV1(Dataset):
    def __init__(
        self,
        data,
        split_name,
        patch_size,
        max_num_patches,
        font_size,
        span_mask_ratio: float = 0.25,
        span_mask_max_length: int = 6,
        span_mask_cum_weights: List[float] = None,
        renderer_name: str = "adaptive",
    ):
        super().__init__()
        self.split_name = split_name
        self.data = data
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.font_size = font_size
        self.font_config = FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)
        if renderer_name == "continuous":
            self.pixel_processor = PixelRendererProcessor(font=self.font_config)
        elif renderer_name == "bigrams":
            self.pixel_processor = BigramsRenderingProcessor(font=self.font_config)
        elif renderer_name == "adaptive":
            self.pixel_processor = AdaptiveRenderingProcessor(font=self.font_config)
        elif renderer_name == "words":
            self.pixel_processor = WordsRenderingProcessor(font=self.font_config)
        else:
            raise ValueError(f"Unknown renderer_name: {renderer_name!r}")
        self.transforms = get_transforms(do_resize=False)

        # Span masking parameters
        self.span_mask_ratio = span_mask_ratio
        self.span_mask_max_length = span_mask_max_length
        self.span_mask_cum_weights = span_mask_cum_weights or [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

        # Initialize span masking generator
        self.mask_generator = SpanMaskingGenerator(
            max_span_length=self.span_mask_max_length,
            spacing='span',
            cumulative_span_weights=self.span_mask_cum_weights,
        )

    def __len__(self):
        return len(self.data)

    def _get_random_data(self):
        max_width = self.max_num_patches * self.patch_size
        out = {
            'pixel_values': torch.rand(3, self.patch_size, max_width),
            'attention_mask': torch.ones(self.max_num_patches),
            'num_text_patches': self.max_num_patches - 1,
            'patch_mask': torch.ones(self.max_num_patches),
        }
        return out

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            processed = preprocess_text_for_pretraining(
                text=item['text'],
                pixel_processor=self.pixel_processor,
                transforms=self.transforms,
                patch_size=self.patch_size,
                max_num_patches=self.max_num_patches,
                mask_generator=self.mask_generator,
                span_mask_ratio=self.span_mask_ratio,
                font_size=self.font_size,
            )
        except Exception:  # sometimes welt's renderer produces errors, so this is why.
            # TODO: Debug this and open an issue
            return self.__getitem__((idx + 1) % len(self))

        # Merge processed data into item
        item.update(processed)
        return item


class FineWebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        hf_iterable_dataset,
        patch_size,
        max_num_patches,
        font_size,
        span_mask_ratio: float = 0.25,
        span_mask_max_length: int = 6,
        span_mask_cum_weights: List[float] = None,
        renderer_name: str = "adaptive",
    ):
        super().__init__()
        self.hf_dataset = hf_iterable_dataset
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.font_size = font_size
        self.font_config = FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)
        if renderer_name == "continuous":
            self.pixel_processor = PixelRendererProcessor(font=self.font_config)
        elif renderer_name == "bigrams":
            self.pixel_processor = BigramsRenderingProcessor(font=self.font_config)
        elif renderer_name == "adaptive":
            self.pixel_processor = AdaptiveRenderingProcessor(font=self.font_config)
        elif renderer_name == "words":
            self.pixel_processor = WordsRenderingProcessor(font=self.font_config)
        else:
            raise ValueError(f"Unknown renderer_name: {renderer_name!r}")
        self.transforms = get_transforms(do_resize=False)
        self.span_mask_ratio = span_mask_ratio
        self.span_mask_max_length = span_mask_max_length
        self.span_mask_cum_weights = span_mask_cum_weights or [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        self.mask_generator = SpanMaskingGenerator(
            max_span_length=self.span_mask_max_length,
            spacing='span',
            cumulative_span_weights=self.span_mask_cum_weights,
        )

    def __iter__(self):
        for item in self.hf_dataset:
            try:
                processed = preprocess_text_for_pretraining(
                    text=item['text'],
                    pixel_processor=self.pixel_processor,
                    transforms=self.transforms,
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    mask_generator=self.mask_generator,
                    span_mask_ratio=self.span_mask_ratio,
                    font_size=self.font_size,
                )
                yield processed
            except Exception:
                continue


class FineWebDataModuleV1(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        font_dir: str = get_fallback_fonts_dir(),
        font_size: int = 12,
        patch_size: int = 16,
        span_mask_ratio: float = 0.25,
        span_mask_max_length: int = 6,
        span_mask_cum_weights: List[float] = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        max_num_patches: int = 512,
        renderer_name: str = "adaptive",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.font_dir = font_dir
        self.font_size = font_size
        self.patch_size = patch_size
        self.span_mask_ratio = span_mask_ratio
        self.span_mask_max_length = span_mask_max_length
        self.span_mask_cum_weights = span_mask_cum_weights
        self.max_num_patches = max_num_patches
        self.renderer_name = renderer_name

    def setup(self, stage: Optional[str]=None):
        """Setup datasets - runs on all processes. Loads from preprocessed Arrow cache."""
        self.ds = load_from_disk(self.data_dir, keep_in_memory=False)
        self.train_dataset = None
        if 'train' in self.ds:
            self.train_dataset = self.setup_split(self.ds['train'], 'train')

    def setup_split(self, dataset, split_name: str):
        return FineWebDatasetV1(
            data=dataset,
            split_name=split_name,
            patch_size=self.patch_size,
            max_num_patches=self.max_num_patches,
            font_size=self.font_size,
            span_mask_ratio=self.span_mask_ratio,
            span_mask_max_length=self.span_mask_max_length,
            span_mask_cum_weights=self.span_mask_cum_weights,
            renderer_name=self.renderer_name,
        )

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        batch = {"pixel_values": pixel_values, "attention_mask": attention_mask}

        # Add patch masks if present
        if "patch_mask" in examples[0]:
            patch_mask = torch.stack([example["patch_mask"] for example in examples])
            batch["patch_mask"] = patch_mask
        return batch

    def base_dataloader(self, dataset, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None
        return self.base_dataloader(self.train_dataset, shuffle=True, drop_last=False)


def interleave_by_lang(files: List[str], seed: int = 42) -> List[str]:
    """Reorder files so that consecutive files come from different languages.

    Groups files by replacing numeric segments in filenames (shard indices) with a
    placeholder — files that differ only in shard number belong to the same group.
    Shuffles within each group, then round-robins across groups.
    """
    rng = random.Random(seed)

    groups: Dict[str, List[str]] = {}
    for f in files:
        key = re.sub(r'\d+', '#', os.path.basename(f))
        groups.setdefault(key, []).append(f)

    for g in groups.values():
        rng.shuffle(g)

    group_lists = list(groups.values())
    rng.shuffle(group_lists)

    result = []
    iters = [iter(g) for g in group_lists]
    while iters:
        remaining = []
        for it in iters:
            val = next(it, None)
            if val is not None:
                result.append(val)
                remaining.append(it)
        iters = remaining

    return result


class FineWebIterableDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        font_size: int = 12,
        patch_size: int = 16,
        span_mask_ratio: float = 0.25,
        span_mask_max_length: int = 6,
        span_mask_cum_weights: List[float] = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        max_num_patches: int = 512,
        train_patterns: List[str] = None,
        renderer_name: str = "adaptive",
        shuffle_buffer_size: int = 100_000,
        seed: int = 42,
        languages: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.font_size = font_size
        self.patch_size = patch_size
        self.span_mask_ratio = span_mask_ratio
        self.span_mask_max_length = span_mask_max_length
        self.span_mask_cum_weights = span_mask_cum_weights
        self.max_num_patches = max_num_patches
        self.train_patterns = train_patterns or ["*-train-*.jsonl.gz", "*-train-*.jsonl.zst"]
        self.renderer_name = renderer_name
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.languages = languages.split(",") if languages else None
        self.train_dataset = None

    def _get_lang_dirs(self) -> List[str]:
        if self.languages:
            return self.languages
        return [d for d in sorted(os.listdir(self.data_dir))
                if os.path.isdir(os.path.join(self.data_dir, d))]

    def _glob_patterns(self, patterns: List[str]) -> List[str]:
        files = []
        for lang_dir in self._get_lang_dirs():
            for pattern in patterns:
                matched = glob.glob(os.path.join(self.data_dir, lang_dir, pattern))
                files.extend(matched)
        return sorted(set(files))

    def setup(self, stage=None):
        train_files = self._glob_patterns(self.train_patterns)
        if not train_files:
            raise FileNotFoundError(
                f"No training files found matching {self.train_patterns} in {self.data_dir}"
            )

        train_files = interleave_by_lang(train_files, seed=self.seed)

        hf_dataset = load_dataset("json", data_files={"train": train_files}, streaming=True)["train"]

        if self.trainer.world_size > 1:
            from datasets.distributed import split_dataset_by_node
            hf_dataset = split_dataset_by_node(
                hf_dataset,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            )

        hf_dataset = hf_dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        self.train_dataset = FineWebIterableDataset(
            hf_iterable_dataset=hf_dataset,
            patch_size=self.patch_size,
            max_num_patches=self.max_num_patches,
            font_size=self.font_size,
            span_mask_ratio=self.span_mask_ratio,
            span_mask_max_length=self.span_mask_max_length,
            span_mask_cum_weights=self.span_mask_cum_weights,
            renderer_name=self.renderer_name,
        )

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        batch = {"pixel_values": pixel_values, "attention_mask": attention_mask}
        if "patch_mask" in examples[0]:
            patch_mask = torch.stack([example["patch_mask"] for example in examples])
            batch["patch_mask"] = patch_mask
        return batch

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
