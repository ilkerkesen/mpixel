from typing import Optional, Any, Dict
import os
import os.path as osp
import torch
from datasets import load_dataset, load_from_disk, get_dataset_config_names
import lightning as L
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from font_download import FontConfig
from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED
from pixel_renderer import PixelRendererProcessor
from mpixel.renderer.processor import (
    AdaptiveRenderingProcessor,
    BigramsRenderingProcessor,
    WordsRenderingProcessor,
)

from mpixel.utils import get_data_dir, Modality
from mpixel.utils.preprocessing import get_attention_mask, get_transforms


HF_DATASET_NAME = "Davlan/sib200"


def _make_renderer(renderer_type: str, font: Optional[FontConfig] = None):
    if renderer_type == "continuous":
        return PixelRendererProcessor(font=font)
    if renderer_type == "bigrams":
        return BigramsRenderingProcessor(font=font)
    if renderer_type == "adaptive":
        return AdaptiveRenderingProcessor(font=font)
    if renderer_type == "words":
        return WordsRenderingProcessor(font=font)
    raise ValueError(f"Unknown renderer_type: {renderer_type!r}")


def preprocess_text_for_classification(
    text: str,
    pixel_processor: PixelRendererProcessor | BigramsRenderingProcessor | AdaptiveRenderingProcessor | WordsRenderingProcessor,
    transforms,
    patch_size: int,
    max_num_patches: int,
    font_size: int,
) -> Dict[str, Any]:
    if hasattr(pixel_processor, "render_text_image"):
        rendered = pixel_processor.render_text_image(
            text=text,
            block_size=patch_size,
            font_size=font_size,
        )
        render_result = None
    else:
        render_result = pixel_processor(
            text=text,
            block_size=patch_size,
            font_size=font_size,
        )
        rendered = Image.fromarray(render_result["pixel_values"])

    img = transforms(rendered)
    c, h, w = img.shape

    # Segmented processors append an explicit black SEP patch. Continuous
    # rendering returns only text pixels, so we add SEP here.
    max_text_width = (max_num_patches - 1) * patch_size
    max_width = max_num_patches * patch_size
    pixel_values = torch.ones((c, h, max_width))

    if render_result is not None:
        text_w = min(max(w - patch_size, 0), max_text_width)
        num_text_patches = text_w // patch_size
        pixel_values[:, :, :text_w] = img[:, :, :text_w]
        sep_start = num_text_patches * patch_size
        if w >= patch_size:
            pixel_values[:, :, sep_start:sep_start + patch_size] = img[:, :, w - patch_size:w]
        else:
            pixel_values[:, :, sep_start:sep_start + patch_size] = 0.0
    else:
        actual_text_w = min(w, max_text_width)
        num_text_patches = actual_text_w // patch_size
        pixel_values[:, :, :actual_text_w] = img[:, :, :actual_text_w]
        sep_start = num_text_patches * patch_size
        pixel_values[:, :, sep_start:sep_start + patch_size] = 0.0

    attention_mask = get_attention_mask(num_text_patches, seq_length=max_num_patches)

    return {
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "num_text_patches": num_text_patches,
    }


class SIB200Dataset(Dataset):
    def __init__(
        self,
        data,
        input_text_key: str,
        label2id: Dict[str, int],
        patch_size: int,
        max_num_patches: int,
        font_size: int,
        renderer_type: str = "adaptive",
    ) -> None:
        super().__init__()
        self.data = data
        self.input_text_key = input_text_key
        self.label2id = label2id
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.font_size = font_size
        self.font_config = FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)
        self.renderer_type = renderer_type
        self.pixel_processor = _make_renderer(renderer_type, font=self.font_config)
        self.transforms = get_transforms(do_resize=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        text = item[self.input_text_key]
        try:
            processed = preprocess_text_for_classification(
                text=text,
                pixel_processor=self.pixel_processor,
                transforms=self.transforms,
                patch_size=self.patch_size,
                max_num_patches=self.max_num_patches,
                font_size=self.font_size,
            )
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        processed["labels"] = torch.tensor(
            self.label2id[item["category"]],
            dtype=torch.long,
        )
        processed["text"] = text
        processed["category"] = item["category"]
        return processed


class SIB200DataModule(L.LightningDataModule):
    def __init__(
        self,
        language: str,
        root_data_dir: str = get_data_dir(),
        num_labels: int = 7,
        batch_size: int = 32,
        num_workers: int = 8,
        max_num_patches: int = 256,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
    ) -> None:
        super().__init__()
        self.language = language
        self.root_data_dir = osp.abspath(osp.expanduser(root_data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_labels = num_labels
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type

    def prepare_data(self) -> None:
        if osp.isdir(self.dataset_dir):
            return
        os.makedirs(self.dataset_dir)
        dataset = load_dataset(HF_DATASET_NAME, self.language)
        dataset.save_to_disk(self.dataset_dir)

    @property
    def dataset_dir(self):
        return osp.join(self.root_data_dir, "sib-200", self.language)

    @property
    def input_text_key(self):
        return "text"

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = load_from_disk(self.dataset_dir)
        self.labels = self.data["train"].unique("category")
        self.labels.sort()
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.train_dataset = self.setup_split(self.data["train"])
        self.val_dataset = self.setup_split(self.data["validation"])
        self.test_dataset = self.setup_split(self.data["test"])

    def setup_split(self, dataset):
        return SIB200Dataset(
            data=dataset,
            input_text_key=self.input_text_key,
            label2id=self.label2id,
            patch_size=self.patch_size,
            max_num_patches=self.max_num_patches,
            font_size=self.font_size,
            renderer_type=self.renderer_type,
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
            "category": [x_i['category'] for x_i in examples],
            "raw_text": [x_i['text'] for x_i in examples],
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

    def test_dataloader(self) -> DataLoader:
        return self.base_dataloader(self.test_dataset)


if __name__ == "__main__":
    all_languages = get_dataset_config_names(HF_DATASET_NAME)
    for language in tqdm(all_languages):
        data_module = SIB200DataModule(language=language)
        data_module.prepare_data()
