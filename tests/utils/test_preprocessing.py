"""Tests for mpixel/utils/preprocessing.py."""

import pickle

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from mpixel.utils.preprocessing import get_transforms


def random_image(mode: str) -> Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, size=(16, 32, 3), dtype=np.uint8)
    return Image.fromarray(pixels).convert(mode)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_transforms_convert_to_rgb(mode):
    """Without resize, the transforms change the image to RGB, then to a tensor."""
    image = random_image(mode)
    expected = ToTensor()(image.convert("RGB"))
    assert torch.equal(get_transforms(do_resize=False)(image), expected)


def test_transforms_can_be_pickled():
    """Spawned data workers (the macOS default) get the transforms through pickle."""
    transforms = get_transforms(do_resize=False)
    image = random_image("RGB")
    restored = pickle.loads(pickle.dumps(transforms))
    assert torch.equal(restored(image), transforms(image))
