import pytest
import json
import gzip
import numpy as np
from unittest.mock import MagicMock
from PIL import Image
from torchvision.transforms import Compose, Lambda, ToTensor
from pixel_renderer import PixelRendererProcessor


@pytest.fixture
def mock_measure_fn():
    """Deterministic measure: one patch per rendering unit."""
    return lambda unit: 1


@pytest.fixture
def mock_pixel_processor():
    """Fake PixelRendererProcessor returning real PIL images sized proportional to text."""
    processor = MagicMock(spec=PixelRendererProcessor)

    def _render(text, block_size=24, font_size=16):
        width = max(block_size, (len(text) * 8 // block_size) * block_size)
        return Image.new("RGB", (width, block_size), color=(255, 255, 255))

    processor.render_text_image = MagicMock(side_effect=_render)
    return processor


@pytest.fixture
def mock_transforms():
    """Lightweight real transforms pipeline (no resize/normalize)."""
    return Compose([
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        ToTensor(),
    ])


@pytest.fixture
def mock_mask_generator():
    """Deterministic mask: first N positions = 1."""
    def _generate(num_patches_total, num_text_patches, num_patches_to_mask):
        mask = np.zeros(num_patches_total, dtype=np.int32)
        mask[:num_patches_to_mask] = 1
        return mask

    return MagicMock(side_effect=_generate)


@pytest.fixture
def create_jsonl_shard(tmp_path):
    """Factory: creates JSONL files in tmp_path/download/."""
    def _create(records, filename, compress=False):
        out_dir = tmp_path / "download"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        opener = gzip.open if compress else open
        kwargs = {"mode": "wt", "encoding": "utf-8"} if compress else {"mode": "w", "encoding": "utf-8"}
        with opener(path, **kwargs) as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path
    return _create
