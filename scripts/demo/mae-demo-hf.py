"""MAE Gradio demo for MPIXEL pretraining models in Hugging Face format.

Same as mae-demo.py, but loads the model and its MPIXELProcessor from a
Hugging Face model directory (e.g. the output of
scripts/utils/convert_to_huggingface.py) instead of a Lightning checkpoint:
text + masking sliders -> original / masked / masked-predictions /
reconstruction images.

Usage:
    python scripts/demo/mae-demo-hf.py --model-dir path/to/hf_model_dir
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
from PIL import Image
from transformers import AutoModelForPreTraining

from mpixel.models import auto_register  # noqa: F401
from mpixel.models.masking import SpanMaskingGenerator
from mpixel.processing import MPIXELProcessor
from mpixel.utils import EXAMPLE_TEXTS


# Span length weights used in pretraining.
SPAN_MASK_CUM_WEIGHTS = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]


def span_patch_mask(attention_mask: torch.Tensor, mask_generator, mask_ratio: float) -> torch.Tensor:
    """Span mask over the text and SEP patches, the same as preprocess_text_for_pretraining."""
    num_patches = int(attention_mask.sum())  # text patches + SEP
    patch_mask = mask_generator(
        num_patches_total=num_patches,
        num_text_patches=num_patches,
        num_patches_to_mask=math.ceil(num_patches * mask_ratio),
    )
    return torch.tensor(np.pad(patch_mask, (0, len(attention_mask) - num_patches), constant_values=0))


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
    processor: MPIXELProcessor,
    span_mask_cum_weights: List[float],
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

    patch_size, max_num_patches = processor.patch_size, processor.max_num_patches
    inputs = processor(text)
    patch_mask = span_patch_mask(inputs["attention_mask"][0], mask_generator, float(mask_ratio))

    pixel_values = inputs["pixel_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    patch_mask = patch_mask.unsqueeze(0).to(device)

    # Override model.config.mask_ratio so the encoder removes exactly the
    # patches our span generator selected. The model's controlled_masking uses
    # `len_keep = seq_length * (1 - config.mask_ratio)` over the full padded
    # sequence; the patch_mask only *ranks* which patches go first. Without
    # this override the slider has almost no effect because the training-time
    # ratio (e.g. 0.25 of 1024) dwarfs the per-text mask budget.
    n_attended = int(inputs["attention_mask"].sum().item())
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
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to a Hugging Face model directory (config.json, preprocessor_config.json + weights).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", default=7860, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    processor = MPIXELProcessor.from_pretrained(args.model_dir)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dtype="auto" keeps the dtype stored in the model directory.
    model = AutoModelForPreTraining.from_pretrained(str(args.model_dir), dtype="auto")
    model.eval()
    model.to(device)

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
            processor=processor,
            span_mask_cum_weights=SPAN_MASK_CUM_WEIGHTS,
            device=device,
        )

    demo = gr.Blocks()
    with demo:
        gr.Markdown("## MPIXEL Masked Autoencoding")
        gr.Markdown(
            f"Gradio demo for the MPIXEL MAE model `{args.model_dir.name}` "
            f"(`{type(model).__name__}`, patch size {processor.patch_size}, "
            f"max patches {processor.max_num_patches}). "
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
