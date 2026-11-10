#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import yaml

from mpixel.models import auto_register  # noqa: F401
from mpixel.processing import MPIXELProcessor
from mpixel.utils.checkpoint import load_model_from_checkpoint


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Lightning checkpoint into a Hugging Face-compatible model directory."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Lightning .ckpt file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the Hugging Face model",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training Lightning config.yaml (for the rendering settings of the processor)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to exported model_config.json (overrides the config stored in the checkpoint)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on missing or unexpected state dict keys (default: on)",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPES),
        default="float32",
        help="Dtype of the saved weights",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation recorded in the saved config",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_cfg = yaml.safe_load(Path(args.config).read_text())["data"]
    if data_cfg["renderer_name"] != "adaptive":
        raise ValueError(f"MPIXELProcessor supports only the adaptive renderer, got {data_cfg['renderer_name']!r}")
    processor = MPIXELProcessor(
        patch_size=data_cfg["patch_size"],
        max_num_patches=data_cfg["max_num_patches"],
        font_size=data_cfg["font_size"],
    )

    model, _, _ = load_model_from_checkpoint(
        str(checkpoint_path),
        attn_implementation=args.attn_implementation,
        inference_dtype=DTYPES[args.dtype],
        model_config_path=args.model_config,
        strict=args.strict,
    )
    print(f"Loaded {type(model).__name__}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
