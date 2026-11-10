from typing import Optional
import os
import os.path as osp
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import lightning as L
from torch.utils.data import DataLoader

from mpixel.utils import get_data_dir
from mpixel.datamodules.glue import GLUEDataset


HF_DATASET_NAME = "xnli"

# Same set/order as the original PIXEL run_xnli_translate_train_all.py.
XNLI_LANGUAGES = ["en", "fr", "es", "el", "de", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]


class XNLIDataModule(L.LightningDataModule):
    """XNLI in the translate-train-all setting: train/validation/test are each
    concatenated across all 15 languages. Premise/hypothesis pairs are rendered
    by the (reused) GLUEDataset."""

    def __init__(
        self,
        languages: list[str] = XNLI_LANGUAGES,
        root_data_dir: str = get_data_dir(),
        num_labels: int = 3,
        batch_size: int = 32,
        num_workers: int = 8,
        max_num_patches: int = 256,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
    ) -> None:
        super().__init__()
        self.languages = languages
        self.root_data_dir = osp.abspath(osp.expanduser(root_data_dir))
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type

    @property
    def sentence_keys(self):
        return "premise", "hypothesis"

    def lang_dir(self, language: str) -> str:
        return osp.join(self.root_data_dir, "xnli", language)

    def prepare_data(self) -> None:
        for language in self.languages:
            lang_dir = self.lang_dir(language)
            if osp.isdir(lang_dir):
                continue
            os.makedirs(lang_dir)
            dataset = load_dataset(HF_DATASET_NAME, language)
            dataset.save_to_disk(lang_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        per_lang = {lang: load_from_disk(self.lang_dir(lang)) for lang in self.languages}

        self.labels = per_lang[self.languages[0]]["train"].features["label"].names
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        # Tag every example with its language (carried through to the batch) so
        # the experiment can bucket per-language accuracy in a single eval pass.
        def concat(split):
            tagged = [
                per_lang[lang][split].add_column("language", [lang] * len(per_lang[lang][split]))
                for lang in self.languages
            ]
            return self.setup_split(concatenate_datasets(tagged))

        self.train_dataset = concat("train")
        self.val_dataset = concat("validation")
        self.test_dataset = concat("test")

    def setup_split(self, dataset):
        sentence1_key, sentence2_key = self.sentence_keys
        return GLUEDataset(
            data=dataset,
            sentence1_key=sentence1_key,
            sentence2_key=sentence2_key,
            is_regression=False,
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
            "language": [example["language"] for example in examples],
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
    data_module = XNLIDataModule()
    data_module.prepare_data()
