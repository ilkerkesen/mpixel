"""Turn text into MPIXEL model inputs (`pixel_values` and `attention_mask`)."""
import json
from pathlib import Path

import torch
from font_download import FontConfig
from huggingface_hub import hf_hub_download
from PIL import Image

from mpixel.renderer.fonts import FONTS_NOTO_SANS_EXTENDED
from mpixel.renderer.processor import AdaptiveRenderingProcessor
from mpixel.utils.preprocessing import get_attention_mask, get_transforms


CONFIG_NAME = "preprocessor_config.json"


class MPIXELProcessor:
    """Renders text into the inputs of a MPIXEL model, the same way as in pretraining.

    Example:
        >>> processor = MPIXELProcessor.from_pretrained("path/or/hub-id")
        >>> inputs = processor(["Hello world!", "Merhaba dünya!"])
        >>> outputs = model(**inputs)
    """

    def __init__(self, patch_size: int = 16, max_num_patches: int = 512, font_size: int = 12):
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.font_size = font_size
        self.renderer = AdaptiveRenderingProcessor(font=FontConfig(sources=FONTS_NOTO_SANS_EXTENDED))
        self.transforms = get_transforms(do_resize=False)

    def __call__(self, texts: str | list[str]) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        pixel_values, attention_mask = zip(*(self._render(text) for text in texts))
        return {"pixel_values": torch.stack(pixel_values), "attention_mask": torch.stack(attention_mask)}

    def _render(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one text into a white-padded strip of `max_num_patches` patches.

        Same as the pretraining pipeline (`preprocess_text_for_pretraining`), without the span masking.
        """
        patch_size = self.patch_size
        rendered = self.renderer(text=text, block_size=patch_size, font_size=self.font_size)
        img = self.transforms(Image.fromarray(rendered["pixel_values"]))
        c, h, w = img.shape

        # The renderer puts a black SEP patch at the end. Keep it, also when the text is cut.
        text_w = min(w - patch_size, (self.max_num_patches - 1) * patch_size)
        pixel_values = torch.ones((c, h, self.max_num_patches * patch_size))
        pixel_values[:, :, :text_w] = img[:, :, :text_w]
        pixel_values[:, :, text_w:text_w + patch_size] = img[:, :, w - patch_size:w]

        attention_mask = get_attention_mask(text_w // patch_size, seq_length=self.max_num_patches)
        return pixel_values, attention_mask

    def to_dict(self) -> dict:
        return {"patch_size": self.patch_size, "max_num_patches": self.max_num_patches, "font_size": self.font_size}

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / CONFIG_NAME).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path) -> "MPIXELProcessor":
        """Load from a local model directory or a Hugging Face Hub model id."""
        local_dir = Path(name_or_path)
        if local_dir.is_dir():
            config_file = local_dir / CONFIG_NAME
        else:
            config_file = hf_hub_download(repo_id=str(name_or_path), filename=CONFIG_NAME)
        return cls(**json.loads(Path(config_file).read_text()))
