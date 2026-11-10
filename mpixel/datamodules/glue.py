from typing import Optional, Any, Dict
import os
import os.path as osp
import torch
from datasets import load_dataset, load_from_disk
import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
from mpixel.datamodules.sib200 import preprocess_text_for_classification, _make_renderer


HF_DATASET_NAME = "glue"

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
    "wnli": 2,
}


def glue_strip_spaces(sent: str):
    """
    Preprocessing function for GLUE inputs
    Naively removes whitespaces before and after certain strings
    """
    sent = sent.replace(" ,", ",")
    sent = sent.replace(" .", ".")
    sent = sent.replace(" !", "!")
    sent = sent.replace(" ?", "?")
    sent = sent.replace(" #", "#")
    sent = sent.replace(" /", "/")
    sent = sent.replace(' "', '"')
    sent = sent.replace('" ', '"')
    sent = sent.replace(" '", "'")
    sent = sent.replace("' ", "'")
    sent = sent.replace(" n't", "n't")
    sent = sent.replace("( ", "(")
    sent = sent.replace(" )", ")")
    sent = sent.replace("[ ", "[")
    sent = sent.replace(" ]", "]")
    return sent


def _render_sentence(text, pixel_processor, transforms, patch_size, font_size):
    if hasattr(pixel_processor, "render_text_image"):
        rendered = pixel_processor.render_text_image(
            text=text, block_size=patch_size, font_size=font_size,
        )
        img = transforms(rendered)
        c, h, w = img.shape
        num_patches = w // patch_size
        return img[:, :, :num_patches * patch_size], num_patches
    else:
        render_result = pixel_processor(
            text=text, block_size=patch_size, font_size=font_size,
        )
        rendered = Image.fromarray(render_result["pixel_values"])
        img = transforms(rendered)
        c, h, w = img.shape
        text_w = max(w - patch_size, 0)
        num_patches = text_w // patch_size
        return img[:, :, :num_patches * patch_size], num_patches


def preprocess_text_pair_for_classification(
    text_a: str,
    text_b: str,
    pixel_processor,
    transforms,
    patch_size: int,
    max_num_patches: int,
    font_size: int,
) -> Dict[str, Any]:
    img_a, patches_a = _render_sentence(text_a, pixel_processor, transforms, patch_size, font_size)
    img_b, patches_b = _render_sentence(text_b, pixel_processor, transforms, patch_size, font_size)

    c, h, _ = img_a.shape
    max_text_patches = max_num_patches - 2

    if patches_a + patches_b > max_text_patches:
        half = max_text_patches // 2
        if patches_a <= half:
            patches_b = max_text_patches - patches_a
        elif patches_b <= half:
            patches_a = max_text_patches - patches_b
        else:
            patches_a = half
            patches_b = max_text_patches - half

    max_width = max_num_patches * patch_size
    pixel_values = torch.ones((c, h, max_width))
    offset = 0

    text_a_w = patches_a * patch_size
    pixel_values[:, :, offset:offset + text_a_w] = img_a[:, :, :text_a_w]
    offset += text_a_w

    pixel_values[:, :, offset:offset + patch_size] = 0.0
    offset += patch_size

    text_b_w = patches_b * patch_size
    pixel_values[:, :, offset:offset + text_b_w] = img_b[:, :, :text_b_w]
    offset += text_b_w

    pixel_values[:, :, offset:offset + patch_size] = 0.0

    # patches_a + 1 (inter-sentence SEP) + patches_b; get_attention_mask adds +1 for the final SEP
    num_text_patches = patches_a + 1 + patches_b
    attention_mask = get_attention_mask(num_text_patches, seq_length=max_num_patches)

    return {
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "num_text_patches": num_text_patches,
    }


class GLUEDataset(Dataset):
    def __init__(
        self,
        data,
        sentence1_key: str,
        sentence2_key: Optional[str],
        is_regression: bool,
        patch_size: int,
        max_num_patches: int,
        font_size: int,
        renderer_type: str = "adaptive",
    ) -> None:
        super().__init__()
        self.data = data
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.is_regression = is_regression
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
        text_a = glue_strip_spaces(" ".join(str(item[self.sentence1_key]).split()))

        try:
            if self.sentence2_key is not None:
                text_b = glue_strip_spaces(" ".join(str(item[self.sentence2_key]).split()))
                processed = preprocess_text_pair_for_classification(
                    text_a=text_a,
                    text_b=text_b,
                    pixel_processor=self.pixel_processor,
                    transforms=self.transforms,
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    font_size=self.font_size,
                )
                text = text_a + " " + text_b
            else:
                text = text_a
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

        label = item["label"]
        if self.is_regression:
            processed["labels"] = torch.tensor(label, dtype=torch.float)
        else:
            processed["labels"] = torch.tensor(label, dtype=torch.long)
        processed["text"] = text
        if "language" in item:
            processed["language"] = item["language"]
        return processed


class GLUEDataModule(L.LightningDataModule):
    def __init__(
        self,
        task_name: str = "sst2",
        root_data_dir: str = get_data_dir(),
        batch_size: int = 32,
        num_workers: int = 8,
        max_num_patches: int = 256,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
    ) -> None:
        super().__init__()
        if task_name not in TASK_TO_KEYS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {list(TASK_TO_KEYS.keys())}"
            )
        self.task_name = task_name
        self.root_data_dir = osp.abspath(osp.expanduser(root_data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type

    @property
    def num_labels(self) -> int:
        return TASK_NUM_LABELS[self.task_name]

    @property
    def is_regression(self) -> bool:
        return self.task_name == "stsb"

    @property
    def sentence_keys(self):
        return TASK_TO_KEYS[self.task_name]

    def prepare_data(self) -> None:
        if osp.isdir(self.dataset_dir):
            return
        os.makedirs(self.dataset_dir)
        dataset = load_dataset(HF_DATASET_NAME, self.task_name)
        dataset.save_to_disk(self.dataset_dir)

    @property
    def dataset_dir(self):
        return osp.join(self.root_data_dir, "glue", self.task_name)

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = load_from_disk(self.dataset_dir)

        self.train_dataset = self.setup_split(self.data["train"])

        if self.task_name == "mnli":
            self.val_dataset = self.setup_split(self.data["validation_matched"])
            # GLUE test labels are hidden; use validation for final eval
            self.test_dataset = self.setup_split(self.data["validation_matched"])
            self.val_mm_dataset = self.setup_split(self.data["validation_mismatched"])
        else:
            self.val_dataset = self.setup_split(self.data["validation"])
            self.test_dataset = self.setup_split(self.data["validation"])

    def setup_split(self, dataset):
        sentence1_key, sentence2_key = self.sentence_keys
        return GLUEDataset(
            data=dataset,
            sentence1_key=sentence1_key,
            sentence2_key=sentence2_key,
            is_regression=self.is_regression,
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
            "raw_text": [x_i["text"] for x_i in examples],
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
    for task in tqdm(list(TASK_TO_KEYS.keys())):
        data_module = GLUEDataModule(task_name=task)
        data_module.prepare_data()
