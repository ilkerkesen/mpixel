"""MAE Gradio demo for MPIXEL pretraining models.

Loads a Lightning checkpoint plus its training config and exposes the same UI
as the original PIXEL Gradio app: text + masking sliders -> original / masked /
masked-predictions / reconstruction images.

Usage:
    python scripts/demo/mae-demo.py \
        --checkpoint path/to/last.ckpt \
        --config     path/to/lightning_logs/.../config.yaml
"""

import argparse
import math
import random
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import torch
import torchvision
import yaml
from PIL import Image

from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor

from mpixel.datamodules.pretraining import preprocess_text_for_pretraining
from mpixel.experiments.pretraining import ModernPixelPretrainingExperiment
from mpixel.models.masking import SpanMaskingGenerator
from mpixel.renderer.processor import (
    AdaptiveRenderingProcessor,
    BigramsRenderingProcessor,
    WordsRenderingProcessor,
)
from mpixel.utils import EXAMPLE_TEXTS, get_root_dir
from mpixel.utils.preprocessing import get_transforms


HEADLESS_VARIANTS = {
    "headlesspixel_v1",
    "headlesspixel_v2",
    "headless_custompixel_v1",
}


def build_pixel_processor(name: str):
    font_config = FontConfig(sources=FONTS_NOTO_SANS)
    if name == "continuous":
        return PixelRendererProcessor(font=font_config)
    if name == "bigrams":
        return BigramsRenderingProcessor(font=font_config)
    if name == "adaptive":
        return AdaptiveRenderingProcessor(font=font_config)
    if name == "words":
        return WordsRenderingProcessor(font=font_config)
    raise ValueError(f"Unknown renderer_name: {name!r}")


def resolve_config_path(saved_path: str, override: Path | None) -> str:
    """Map the yaml's training-host config_path to a local one.

    Lightning's load_from_checkpoint reuses the saved hparams, including the
    absolute path captured at training time. Look for a same-basename file
    under the repo's configs/ directory (or accept an explicit override).
    """
    if override is not None:
        return str(override)
    candidate = Path(get_root_dir()) / "configs" / Path(saved_path).name
    if candidate.exists():
        return str(candidate)
    return saved_path


def to_image(img_tensor: torch.Tensor, do_clip: bool) -> Image.Image:
    """Convert a (C, H, W) patch strip to a square-ish PIL grid image.

    Mirrors the old PIXEL demo: clip+scale for already-pixel inputs (originals
    and the masked view), let make_grid(normalize=True) handle predictions and
    reconstructions whose values can land outside [0, 1].
    """
    img = img_tensor.detach().cpu().float()
    if do_clip:
        img = torch.einsum("chw->hwc", img)
        img = torch.clip(img * 255, 0, 255)
        img = torch.einsum("hwc->chw", img)

    grid = torchvision.utils.make_grid(img.unsqueeze(0), normalize=True)
    grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    h, w, c = grid.shape  # h == patch_size, w == num_patches * patch_size

    num_patches = w // h
    grid_size = int(math.ceil(math.sqrt(num_patches)))
    if num_patches < grid_size ** 2:
        pad_n = grid_size ** 2 - num_patches
        pad = np.full((h, pad_n * h, c), 255, dtype=np.uint8)
        grid = np.concatenate([grid, pad], axis=1)

    grid = (
        grid.reshape(h, grid_size, grid_size, h, c)
        .transpose(1, 0, 2, 3, 4)
        .reshape(grid_size * h, grid_size * h, c)
    )
    return Image.fromarray(grid)


@torch.no_grad()
def run_inference(
    text: str,
    mask_ratio: float,
    max_span_length: int,
    seed: int,
    *,
    model,
    pixel_processor,
    transforms,
    patch_size: int,
    max_num_patches: int,
    span_mask_cum_weights: List[float],
    font_size: int,
    device: torch.device,
):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mask_generator = SpanMaskingGenerator(
        max_span_length=int(max_span_length),
        spacing="span",
        cumulative_span_weights=list(span_mask_cum_weights[: int(max_span_length)]),
    )

    processed = preprocess_text_for_pretraining(
        text=text,
        pixel_processor=pixel_processor,
        transforms=transforms,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        mask_generator=mask_generator,
        span_mask_ratio=float(mask_ratio),
        font_size=font_size,
    )

    pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
    attention_mask = processed["attention_mask"].unsqueeze(0).to(device)
    patch_mask = processed["patch_mask"].unsqueeze(0).to(device)

    # Override model.config.mask_ratio so the encoder removes exactly the
    # patches our span generator selected. The model's controlled_masking uses
    # `len_keep = seq_length * (1 - config.mask_ratio)` over the full padded
    # sequence; the patch_mask only *ranks* which patches go first. Without
    # this override the slider has almost no effect because the training-time
    # ratio (e.g. 0.25 of 1024) dwarfs the per-text mask budget.
    n_attended = int(processed["attention_mask"].sum().item())
    num_to_mask = math.ceil(n_attended * float(mask_ratio))
    prev_mask_ratio = model.config.mask_ratio
    model.config.mask_ratio = num_to_mask / float(max_num_patches)

    # Pretraining ran with bf16-mixed precision; flash_attention_2 also requires
    # fp16/bf16 activations, so mirror that here for CUDA. Skip on CPU/MPS where
    # autocast is unnecessary (and bf16 flash kernels aren't available).
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    try:
        with autocast_ctx:
            output = model(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                patch_mask=patch_mask,
                return_dict=True,
            )
    finally:
        model.config.mask_ratio = prev_mask_ratio

    predictions = model.unpatchify(output.logits.float()).squeeze(0).detach().cpu()

    mask = output.mask.detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
    mask = model.unpatchify(mask).squeeze(0)

    attn_full = attention_mask.detach().cpu().unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
    attn_full = model.unpatchify(attn_full).squeeze(0)

    original = model.unpatchify(model.patchify(pixel_values)).squeeze(0).detach().cpu()

    visible = (torch.bitwise_and(mask == 1, attn_full == 1)).long()
    im_masked = original * (1 - visible)
    masked_predictions = predictions * mask * attn_full
    reconstruction = im_masked + masked_predictions

    return [
        to_image(original, do_clip=True),
        to_image(im_masked, do_clip=True),
        to_image(masked_predictions, do_clip=False),
        to_image(reconstruction, do_clip=False),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a Lightning .ckpt file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the training Lightning config.yaml.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional override for model.config_path (architecture JSON). "
             "If unset, the script tries configs/<basename> under the repo root.",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", default=7860, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    model_variant = model_cfg.get("model_variant", "custompixel_v1")
    if model_variant in HEADLESS_VARIANTS:
        raise ValueError(
            f"Headless variants (got {model_variant!r}) don't expose decoder reconstructions; "
            "use a non-headless pretraining checkpoint."
        )

    renderer_name = data_cfg.get("renderer_name", "continuous")
    patch_size = int(data_cfg.get("patch_size", 24))
    max_num_patches = int(data_cfg.get("max_num_patches", 529))
    font_size = int(data_cfg.get("font_size", 16))
    span_mask_cum_weights = data_cfg.get("span_mask_cum_weights") or [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    pixel_processor = build_pixel_processor(renderer_name)
    transforms = get_transforms(do_resize=False)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_config_path = model_cfg.get("config_path", "")
    local_config_path = resolve_config_path(saved_config_path, args.config_path)

    experiment = ModernPixelPretrainingExperiment.load_from_checkpoint(
        str(args.checkpoint),
        map_location=device,
        config_path=local_config_path,
    )
    experiment.eval()
    experiment.to(device)
    model = experiment.model

    # The pretraining model compiles the embedding step (`_compiled_embeddings`)
    # which captures `config.mask_ratio` into the graph. We mutate that ratio
    # per inference call to honour the slider, so disable the compile path.
    if hasattr(model, "config") and getattr(model.config, "reference_compile", False):
        model.config.reference_compile = False

    def inference(text, mask_ratio, max_span_length, seed):
        return run_inference(
            text,
            mask_ratio,
            max_span_length,
            seed,
            model=model,
            pixel_processor=pixel_processor,
            transforms=transforms,
            patch_size=patch_size,
            max_num_patches=max_num_patches,
            span_mask_cum_weights=span_mask_cum_weights,
            font_size=font_size,
            device=device,
        )

    demo = gr.Blocks()
    with demo:
        gr.Markdown("## MPIXEL Masked Autoencoding")
        gr.Markdown(
            f"Gradio demo for the MPIXEL MAE checkpoint `{args.checkpoint.name}` "
            f"(variant `{model_variant}`, renderer `{renderer_name}`, "
            f"patch size {patch_size}, max patches {max_num_patches}). "
            "Enter text or pick an example, then hit **Run** to render, mask, and reconstruct."
        )
        with gr.Row():
            with gr.Column():
                tb_text = gr.Textbox(label="Text", lines=4)
                sl_ratio = gr.Slider(0.01, 1.0, step=0.01, value=0.25, label="Span masking ratio")
                sl_len = gr.Slider(1, 6, step=1, value=6, label="Masking max span length")
                sl_seed = gr.Slider(0, 1000, step=1, value=42, label="Random seed")
                btn = gr.Button("Run")
            with gr.Column():
                out_original = gr.Image(label="Original")
                out_masked = gr.Image(label="Masked")
                out_pred = gr.Image(label="Masked Predictions")
                out_reconstruction = gr.Image(label="Reconstruction")

        gr.Examples(
            examples=[[t, 0.25, 6, 42] for t in EXAMPLE_TEXTS],
            inputs=[tb_text, sl_ratio, sl_len, sl_seed],
        )

        btn.click(
            fn=inference,
            inputs=[tb_text, sl_ratio, sl_len, sl_seed],
            outputs=[out_original, out_masked, out_pred, out_reconstruction],
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
