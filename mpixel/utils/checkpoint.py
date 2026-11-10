"""Utilities for loading models from PyTorch Lightning checkpoints."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers.utils import SAFE_WEIGHTS_NAME

from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining


def load_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract and clean state dict from a Lightning checkpoint.

    Strips 'model.' prefix (Lightning) and '_orig_mod.' prefix (torch.compile).
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_pretrained_state_dict(model_id_or_path: str) -> Dict[str, torch.Tensor]:
    """Load the weights of a pretrained model from a Lightning .ckpt file, or from an HF model folder or Hub ID
    (the output of scripts/utils/convert_to_huggingface.py). Both give the same parameter names."""
    if model_id_or_path.endswith(".ckpt"):
        return load_state_dict(torch.load(model_id_or_path, map_location="cpu", weights_only=False))
    model_dir = Path(model_id_or_path)
    if not model_dir.is_dir():
        model_dir = Path(snapshot_download(model_id_or_path, allow_patterns=[SAFE_WEIGHTS_NAME]))
    weights_path = model_dir / SAFE_WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{model_id_or_path} has no {SAFE_WEIGHTS_NAME}. Make the HF model folder with "
            "scripts/utils/convert_to_huggingface.py."
        )
    return load_file(weights_path)


def load_hf_model_config(model_id_or_path: str, **kwargs) -> MPIXELConfig:
    """Load the config of an HF model folder or Hub ID (scripts/utils/convert_to_huggingface.py) for fine-tuning.

    The conversion turns off reference_compile. Turn it on again, the same as in the Lightning checkpoints of the
    reported runs. kwargs replace config values, for example num_labels.
    """
    config = MPIXELConfig.from_pretrained(model_id_or_path, **kwargs)
    config.reference_compile = True
    return config


def config_from_checkpoint(model_config: Dict[str, Any]) -> MPIXELConfig:
    """Build an MPIXELConfig from the 'model_config' entry of a Lightning checkpoint.

    The release checkpoints were saved by the old code, so their model_config has
    model_type 'pixel_next' and settings of removed features. Ignore both.
    """
    known_keys = MPIXELConfig().to_dict().keys() - {"model_type"}
    return MPIXELConfig(**{k: v for k, v in model_config.items() if k in known_keys and not k.startswith("_")})


def load_model_from_checkpoint(
    checkpoint_path: str,
    attn_implementation: str = "sdpa",
    inference_dtype: torch.dtype = torch.float32,
    model_config_path: Optional[str] = None,
    strict: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    """Load a model from a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        attn_implementation: Attention implementation ('eager' or 'sdpa').
        inference_dtype: Dtype for inference (default float32 for numerical stability).
        model_config_path: Optional model_config.json that overrides the config stored in the checkpoint.
        strict: Raise an error on missing or unexpected state dict keys instead of printing a warning.

    Returns:
        Tuple of (model, encoder, hyperparameters).
        - model: The full pretrained model.
        - encoder: The encoder submodule (for extracting hidden states).
        - hparams: Hyperparameters from the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hparams = ckpt.get("hyper_parameters", {})
    model_cls, config_cls, encoder_attr = MPIXELForPreTraining, MPIXELConfig, 'vit'

    if model_config_path is not None:
        config_dict = json.loads(Path(model_config_path).read_text())
    else:
        config_dict = ckpt.get("model_config")
    if config_dict is None:
        raise RuntimeError(
            "Checkpoint has no 'model_config' entry; pass model_config_path to rebuild the model."
        )

    known_keys = config_cls().to_dict().keys() - {"model_type"}
    config_dict = {k: v for k, v in config_dict.items() if k in known_keys}
    config_dict["reference_compile"] = False
    config_dict["attn_implementation"] = attn_implementation
    config_dict["dtype"] = inference_dtype
    config = config_cls(**{k: v for k, v in config_dict.items() if not k.startswith("_")})
    config._attn_implementation = attn_implementation

    model = model_cls._from_config(config)
    model = model.to(inference_dtype)

    state_dict = load_state_dict(ckpt)
    result = model.load_state_dict(state_dict, strict=strict)
    if result.missing_keys:
        print(f"[warn] missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"[warn] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")

    encoder = getattr(model, encoder_attr)
    return model, encoder, hparams
