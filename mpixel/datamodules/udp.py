import glob
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import lightning as L
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from transformers.file_utils import ModelOutput

from font_download import FontConfig
from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED

from mpixel.utils import Modality, get_data_dir
from mpixel.utils.preprocessing import get_attention_mask, get_transforms
from mpixel.renderer.functional import _render_units
from mpixel.renderer.processor import (
    AdaptiveRenderingProcessor,
    BigramsRenderingProcessor,
    WordsRenderingProcessor,
    _segments_to_units,
)

logger = logging.getLogger(__name__)

_RTL_TREEBANK_LANGUAGES = {
    "Ancient_Hebrew",
    "Arabic",
    "Assyrian",
    "Hebrew",
    "Khunsari",
    "Nayini",
    "Persian",
    "Soi",
    "South_Levantine_Arabic",
    "Urdu",
    "Uyghur",
}


UPOS_LABELS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


class Split(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class DependencyParsingModelOutput(ModelOutput):
    """
    Class for outputs of dependency parsing models.
    """

    loss: Optional[torch.FloatTensor] = None
    arc_logits: torch.FloatTensor = None
    rel_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class UDInputExample:
    """
    A single training/test example for universal dependency parsing.
    """

    words: List[str]
    head_labels: Optional[List[int]]
    rel_labels: Optional[List[str]]


def get_file(data_dir: str, mode: Union[Split, str]) -> Optional[str]:
    if isinstance(mode, Split):
        mode = mode.value
    fp = os.path.join(data_dir, f"*-ud-{mode}.conllu")
    _fp = glob.glob(fp)
    if len(_fp) == 1:
        return _fp[0]
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[UDInputExample]:
    file_path = get_file(data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        head_labels: List[int] = []
        rel_labels: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(UDInputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
                    words = []
                    head_labels = []
                    rel_labels = []
            if tok[0].isdigit():
                word, head, label = tok[1], tok[6], tok[7]
                words.append(word)
                head_labels.append(int(head))
                rel_labels.append(label.split(":")[0])
        if words:
            examples.append(UDInputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
    return examples


def _treebank_language(data_dir: str) -> str:
    treebank = os.path.basename(os.path.normpath(data_dir))
    if treebank.startswith("UD_"):
        treebank = treebank[3:]
    return treebank.split("-", 1)[0]


def _infer_text_direction(data_dir: str) -> str:
    return "rtl" if _treebank_language(data_dir) in _RTL_TREEBANK_LANGUAGES else "ltr"


def _word_order_for_direction(num_words: int, text_direction: str) -> List[int]:
    if text_direction == "rtl":
        return list(range(num_words - 1, -1, -1))
    if text_direction == "ltr":
        return list(range(num_words))
    raise ValueError(f"Unknown text_direction: {text_direction!r}")


def _remap_labels_to_visual_order(
    example: UDInputExample,
    visual_to_logical: List[int],
    n_words: int,
    label_map: Dict[str, int],
    pad_token: int,
) -> Tuple[List[int], List[int]]:
    kept_visual_to_logical = visual_to_logical[:n_words]
    logical_to_visual = {
        logical_idx: visual_idx
        for visual_idx, logical_idx in enumerate(kept_visual_to_logical)
    }

    arc_labels: List[int] = []
    rel_labels: List[int] = []
    for logical_idx in kept_visual_to_logical:
        head = example.head_labels[logical_idx]
        if head == 0:
            arc_labels.append(0)
        elif 1 <= head <= len(example.words) and (head - 1) in logical_to_visual:
            arc_labels.append(logical_to_visual[head - 1] + 1)
        else:
            arc_labels.append(pad_token)

        rel_labels.append(label_map[example.rel_labels[logical_idx]])

    return arc_labels, rel_labels


def _word_units_for_renderer(word: str, renderer_type: str) -> List[str]:
    if renderer_type == "bigrams":
        units, _ = _segments_to_units(word, strategy="bigrams")
    elif renderer_type == "adaptive":
        units, _ = _segments_to_units(word, strategy="auto")
    elif renderer_type == "words":
        units, _ = _segments_to_units(word, strategy="words")
    else:
        raise ValueError(f"Unknown renderer_type: {renderer_type!r}")
    return units or [word]


def _make_renderer(renderer_type: str, font: Optional[FontConfig] = None):
    if renderer_type == "bigrams":
        return BigramsRenderingProcessor(font=font)
    elif renderer_type == "adaptive":
        return AdaptiveRenderingProcessor(font=font)
    elif renderer_type == "words":
        return WordsRenderingProcessor(font=font)
    else:
        raise ValueError(f"Unknown renderer_type: {renderer_type!r}")


class UDDataset(Dataset):
    def __init__(
        self,
        examples: List[UDInputExample],
        label_list: List[str],
        modality: Modality,
        max_num_patches: int = 512,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
        text_direction: str = "ltr",
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        pad_token: int = -100,
    ):
        self.examples = examples
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.modality = modality
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type
        self.text_direction = text_direction
        self.pad_token = pad_token

        if modality == Modality.IMAGE:
            font_config = FontConfig(sources=FONTS_NOTO_SANS_EXTENDED)
            self.renderer = _make_renderer(renderer_type, font=font_config)
            self.transforms = get_transforms(do_resize=False)
        else:
            assert tokenizer is not None, "tokenizer is required for text modality"
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        try:
            if self.modality == Modality.IMAGE:
                return self._process_image(example)
            else:
                return self._process_text(example)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    def _process_image(self, example: UDInputExample) -> Dict[str, Any]:
        if self.renderer._font_owner is not None:
            self.renderer._font_owner._ensure_fontconfig_initialized()

        # UDP uses one direction for the whole treebank. For RTL treebanks,
        # reverse CoNLL token order; ignore mixed-script spans inside the
        # sentence because dependency labels are defined over UD tokens.
        visual_to_logical = _word_order_for_direction(len(example.words), self.text_direction)
        visual_words = [example.words[i] for i in visual_to_logical]
        visual_units: List[str] = []
        unit_counts_per_word: List[int] = []
        for word in visual_words:
            word_units = _word_units_for_renderer(word, self.renderer_type)
            visual_units.extend(word_units)
            unit_counts_per_word.append(len(word_units))

        image_np, patch_counts = _render_units(
            visual_units,
            block_size=self.patch_size,
            font_size=self.font_size,
            pad=1,
        )
        # Append black SEP patch (matches the high-level processors' contract).
        sep_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        image_np = np.concatenate([image_np, sep_patch], axis=1)

        # word_starts_all[i] = first patch index of visual_words[i].
        word_starts_all: List[int] = []
        cursor = 0
        patch_offset = 0
        for unit_count in unit_counts_per_word:
            word_starts_all.append(cursor)
            word_patch_count = sum(patch_counts[patch_offset:patch_offset + unit_count])
            cursor += word_patch_count
            patch_offset += unit_count
        num_text_patches_rendered = cursor

        max_text_patches = self.max_num_patches - 1   # reserve last slot for SEP
        num_text_patches = min(num_text_patches_rendered, max_text_patches)

        # Drop trailing words whose patch span doesn't fully fit before SEP.
        boundaries = word_starts_all + [num_text_patches_rendered]
        n_words = 0
        while n_words < len(word_starts_all) and boundaries[n_words + 1] <= num_text_patches:
            n_words += 1
        word_starts = word_starts_all[:n_words]
        num_text_patches = boundaries[n_words] if n_words > 0 else 0

        # Build padded strip (white bg + text + SEP + padding).
        img_tensor = self.transforms(Image.fromarray(image_np))  # (C, H, W_with_sep)
        c, h = img_tensor.shape[0], img_tensor.shape[1]
        max_width = self.max_num_patches * self.patch_size
        pixel_values = torch.ones((c, h, max_width))
        text_width = num_text_patches * self.patch_size
        pixel_values[:, :, :text_width] = img_tensor[:, :, :text_width]
        pixel_values[:, :, text_width:text_width + self.patch_size] = 0.0

        attention_mask = get_attention_mask(num_text_patches, seq_length=self.max_num_patches)

        pad = [self.pad_token]

        arc_labels, rel_labels = _remap_labels_to_visual_order(
            example,
            visual_to_logical,
            n_words,
            self.label_map,
            self.pad_token,
        )
        arc_labels += (self.max_num_patches - len(arc_labels)) * pad

        rel_labels += (self.max_num_patches - len(rel_labels)) * pad

        # num_text_patches is the end-sentinel for the last word; pad the rest
        word_starts_padded = word_starts + [num_text_patches] + (self.max_num_patches - len(word_starts)) * pad

        return {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "word_starts": torch.tensor(word_starts_padded, dtype=torch.long),
            "arc_labels": torch.tensor(arc_labels, dtype=torch.long),
            "rel_labels": torch.tensor(rel_labels, dtype=torch.long),
        }

    def _process_text(self, example: UDInputExample) -> Dict[str, Any]:
        max_seq_length = self.max_num_patches

        tokens = [self.tokenizer.tokenize(w) for w in example.words]
        word_lengths = [len(w) for w in tokens]

        encoding = self.tokenizer(
            example.words,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        pad = [self.pad_token]
        # cumsum([1] + word_lengths) gives first-token position of each word (1-indexed for [CLS])
        word_starts = [w for w in np.cumsum([1] + word_lengths).tolist() if w < max_seq_length]

        arc_labels = example.head_labels[:len(word_starts)]
        arc_labels = [self.pad_token if al > max_seq_length else al for al in arc_labels]
        rel_labels = [self.label_map[r] for r in example.rel_labels[:len(word_starts)]]

        arc_labels += (max_seq_length - len(arc_labels)) * pad
        rel_labels += (max_seq_length - len(rel_labels)) * pad
        word_starts += (max_seq_length + 1 - len(word_starts)) * pad

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "word_starts": torch.tensor(word_starts, dtype=torch.long),
            "arc_labels": torch.tensor(arc_labels, dtype=torch.long),
            "rel_labels": torch.tensor(rel_labels, dtype=torch.long),
        }


class UDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        modality: Modality = Modality.IMAGE,
        labels: List[str] = UD_HEAD_LABELS,
        batch_size: int = 32,
        num_workers: int = 8,
        max_num_patches: int = 512,
        patch_size: int = 16,
        font_size: int = 12,
        renderer_type: str = "adaptive",
        text_direction: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        pad_token: int = -100,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.modality = modality
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.font_size = font_size
        self.renderer_type = renderer_type
        self.text_direction = text_direction or _infer_text_direction(data_dir)
        self.tokenizer_name = tokenizer_name
        self.pad_token = pad_token

    def setup(self, stage: Optional[str] = None) -> None:
        tokenizer = None
        if self.modality == Modality.TEXT:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.train_dataset = self._setup_split(Split.TRAIN, tokenizer)
        self.val_dataset = self._setup_split(Split.DEV, tokenizer)
        self.test_dataset = self._setup_split(Split.TEST, tokenizer)

    def _setup_split(self, split: Split, tokenizer=None) -> Optional[UDDataset]:
        if get_file(self.data_dir, split) is None:
            return None
        examples = read_examples_from_file(self.data_dir, split)
        return UDDataset(
            examples=examples,
            label_list=self.labels,
            modality=self.modality,
            max_num_patches=self.max_num_patches,
            patch_size=self.patch_size,
            font_size=self.font_size,
            renderer_type=self.renderer_type,
            text_direction=self.text_direction,
            tokenizer=tokenizer,
            pad_token=self.pad_token,
        )

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in examples[0]:
            if isinstance(examples[0][key], torch.Tensor):
                batch[key] = torch.stack([ex[key] for ex in examples])
            else:
                batch[key] = [ex[key] for ex in examples]
        return batch

    def _base_dataloader(self, dataset: UDDataset, shuffle: bool = False) -> DataLoader:
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
        return self._base_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._base_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._base_dataloader(self.test_dataset)
