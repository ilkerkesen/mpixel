"""Utilities for extracting embeddings from PIXEL models."""
from typing import Dict, Tuple

import numpy as np
import torch


def get_extended_mask(attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Left-pad attention mask with 1s to match any prefix tokens (CLS, etc.)."""
    prefix = seq_len - attention_mask.shape[1]
    if prefix > 0:
        pad = torch.ones(attention_mask.shape[0], prefix, device=attention_mask.device, dtype=attention_mask.dtype)
        return torch.cat([pad, attention_mask], dim=1)
    return attention_mask[:, :seq_len]


@torch.no_grad()
def forward_batch(
    encoder: torch.nn.Module,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """Forward a batch through encoder and return hidden states.

    Args:
        encoder: The encoder module (e.g., model.vit).
        pixel_values: Input tensor of shape (B, C, H, W).
        attention_mask: Attention mask of shape (B, seq_len).

    Returns:
        Tuple of (hidden_states, attention_mask).
        hidden_states is a tuple of tensors, one per layer.
    """
    outputs = encoder(
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model returned no hidden_states — check that output_hidden_states is wired.")
    return hidden_states, attention_mask


@torch.no_grad()
def embed_batch(
    encoder: torch.nn.Module,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[int, np.ndarray]:
    """Forward a batch and return mean-pooled embeddings per layer.

    Args:
        encoder: The encoder module.
        pixel_values: Input tensor of shape (B, C, H, W).
        attention_mask: Attention mask of shape (B, seq_len).

    Returns:
        Dict mapping layer index to (B, D) numpy array of embeddings.
    """
    hidden_states, attention_mask = forward_batch(encoder, pixel_values, attention_mask)

    embeddings = {}
    for layer_idx, h in enumerate(hidden_states):
        ext_mask = get_extended_mask(attention_mask, h.shape[1]).unsqueeze(-1).to(h.dtype)
        pooled = (h * ext_mask).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        embeddings[layer_idx] = pooled.float().cpu().numpy()
    return embeddings


@torch.no_grad()
def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool hidden states using attention mask.

    Args:
        hidden_states: Tensor of shape (B, seq_len, D).
        attention_mask: Attention mask of shape (B, seq_len).

    Returns:
        Pooled tensor of shape (B, D).
    """
    ext_mask = get_extended_mask(attention_mask, hidden_states.shape[1]).unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * ext_mask).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden_states.dtype)
    return pooled
