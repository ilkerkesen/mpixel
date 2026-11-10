"""Visualization utilities for pretraining reconstruction."""

from typing import List, Tuple
import math

import torch
import torchvision
import numpy as np

from mpixel.datamodules.pretraining import preprocess_text_for_pretraining


def patches_to_grid_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert linear patches to 2D grid for visualization.

    Args:
        img_tensor: Image tensor of shape (C, H, W) where W = num_patches * patch_size.

    Returns:
        Numpy array of shape (grid_H, grid_W, C) in uint8 format.
    """
    # Clip and normalize
    img = torch.einsum("chw->hwc", img_tensor)
    img = torch.clip(img * 255, 0, 255)
    img = torch.einsum("hwc->chw", img)

    # Make grid
    img = torchvision.utils.make_grid(img.unsqueeze(0), normalize=True)
    img = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    H, W, C = img.shape  # H=patch_size, W=num_patches*patch_size, C=3

    # Reshape into 2D grid (square-ish layout)
    num_patches = W // H
    grid_size = int(math.ceil(math.sqrt(num_patches)))

    # Pad if needed to make square grid
    if num_patches < grid_size ** 2:
        padding_patches = grid_size ** 2 - num_patches
        padding = torch.ones(C, H, padding_patches * H) * 255
        img_padded = torch.cat([
            torch.from_numpy(img).permute(2, 0, 1).float(),
            padding
        ], dim=2)
        img = img_padded.permute(1, 2, 0).numpy().astype(np.uint8)
        W = grid_size * H

    # Reshape to grid
    img_reshaped = (
        img.reshape(H, grid_size, grid_size, H, C)
        .transpose(1, 0, 2, 3, 4)
        .reshape(grid_size * H, grid_size * H, C)
    )

    return img_reshaped


def create_reconstruction_visualization(
    model,
    text: str,
    pixel_processor,
    transforms,
    mask_generator,
    patch_size: int,
    max_num_patches: int,
    span_mask_ratio: float,
    font_size: int,
    device: torch.device,
) -> np.ndarray:
    """Create reconstruction visualization for a single text example.

    Args:
        model: The pretraining model with patchify/unpatchify methods.
        text: The text to render and reconstruct.
        pixel_processor: Processor for rendering text to pixels.
        transforms: Image transforms to apply.
        mask_generator: Generator for creating masks.
        patch_size: Size of each patch.
        max_num_patches: Maximum number of patches.
        span_mask_ratio: Ratio for span masking.
        font_size: Font size for rendering.
        device: Device to run inference on.

    Returns:
        Reconstruction visualization as numpy array.
    """
    # Preprocess text
    processed = preprocess_text_for_pretraining(
        text=text,
        pixel_processor=pixel_processor,
        transforms=transforms,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        mask_generator=mask_generator,
        span_mask_ratio=span_mask_ratio,
        font_size=font_size,
    )

    pixel_values = processed['pixel_values']
    attention_mask = processed['attention_mask']
    patch_mask = processed['patch_mask']

    # Prepare batch (add batch dimension and move to device)
    batch = {
        'pixel_values': pixel_values.unsqueeze(0).to(device),
        'attention_mask': attention_mask.unsqueeze(0).to(device),
        'patch_mask': patch_mask.unsqueeze(0).to(device),
    }

    # Forward pass
    output = model(
        pixel_values=batch['pixel_values'],
        attention_mask=batch['attention_mask'],
        patch_mask=batch['patch_mask'],
        return_dict=True,
    )

    # Unpatchify predictions
    predictions = model.unpatchify(output.logits).detach().cpu().squeeze()

    # Unpatchify mask
    mask = output.mask.detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
    mask = model.unpatchify(mask).squeeze()

    # Unpatchify attention mask
    attn_mask = attention_mask.unsqueeze(-1).repeat(1, patch_size ** 2 * 3)
    attn_mask = model.unpatchify(attn_mask.unsqueeze(0)).squeeze()

    # Get original image (patchify then unpatchify to match model processing)
    original_img = model.unpatchify(
        model.patchify(pixel_values.unsqueeze(0).to(device))
    ).squeeze().detach().cpu()

    # Create masked image (original with masked regions removed)
    im_masked = original_img * (1 - (torch.bitwise_and(mask == 1, attn_mask == 1)).long())

    # Create masked predictions (only the predicted masked regions)
    masked_predictions = predictions * mask * attn_mask

    # Create reconstruction (masked image + masked predictions)
    reconstruction = im_masked + masked_predictions

    return patches_to_grid_image(reconstruction)


def generate_reconstruction_visualizations(
    model,
    texts: List[str],
    pixel_processor,
    transforms,
    mask_generator,
    patch_size: int,
    max_num_patches: int,
    span_mask_ratio: float,
    font_size: int,
    device: torch.device,
) -> List[Tuple[int, np.ndarray]]:
    """Generate reconstruction visualizations for multiple texts.

    Args:
        model: The pretraining model with patchify/unpatchify methods.
        texts: List of texts to render and reconstruct.
        pixel_processor: Processor for rendering text to pixels.
        transforms: Image transforms to apply.
        mask_generator: Generator for creating masks.
        patch_size: Size of each patch.
        max_num_patches: Maximum number of patches.
        span_mask_ratio: Ratio for span masking.
        font_size: Font size for rendering.
        device: Device to run inference on.

    Returns:
        List of (index, image) tuples. Failed examples are skipped.
    """
    results = []
    for idx, text in enumerate(texts):
        try:
            vis = create_reconstruction_visualization(
                model=model,
                text=text,
                pixel_processor=pixel_processor,
                transforms=transforms,
                mask_generator=mask_generator,
                patch_size=patch_size,
                max_num_patches=max_num_patches,
                span_mask_ratio=span_mask_ratio,
                font_size=font_size,
                device=device,
            )
            results.append((idx, vis))
        except Exception as e:
            print(f"Failed to process example {idx}: {e}")
            continue
    return results
