"""
Unified Gradio demo for all three rendering processors. Pick a processor,
type text, and see the rendered strip with patch boundaries overlaid.

Segment starts (segment_start_mask == 1) are drawn in red, other patch
boundaries in light gray.

Usage:
    pip install gradio
    python scripts/demo/renderer.py
"""

import gradio as gr
import numpy as np

from mpixel.renderer.processor import (
    AdaptiveRenderingProcessor,
    BigramsRenderingProcessor,
    WordsRenderingProcessor,
    _GRAPHEME_RE,
    _is_whitespace,
    _segment_graphemes,
)


_RED = np.array([220, 40, 40], dtype=np.uint8)
_GRAY = np.array([200, 200, 200], dtype=np.uint8)

_PROCESSORS = {
    "bigrams": BigramsRenderingProcessor(strategy="bigrams"),
    "adaptive": AdaptiveRenderingProcessor(),
    "words": WordsRenderingProcessor(),
}


def _list_segments(text: str) -> list[str]:
    graphemes = _GRAPHEME_RE.findall(text)
    return [s for s in _segment_graphemes(graphemes) if not _is_whitespace(s)]


def _draw_patch_boundaries(image: np.ndarray, block_size: int, mask: np.ndarray) -> np.ndarray:
    """Overlay vertical lines at every patch boundary. Segment starts are red,
    every other boundary is gray. The final right edge is always gray."""
    h, w, _ = image.shape
    if w == 0:
        return image
    out = image.copy()
    for i, flag in enumerate(mask):
        x = i * block_size
        if x < w:
            out[:, x, :] = _RED if flag == 1 else _GRAY
    out[:, w - 1, :] = _GRAY
    return out


def render(
    text: str,
    processor_name: str,
    block_size: int,
    font_size: int,
    pad_left: int,
    pad_right: int,
    upscale: int,
    mirror: bool,
):
    block_size = int(block_size)
    font_size = int(font_size)
    upscale = int(upscale)
    pad = (int(pad_left), int(pad_right))

    processor = _PROCESSORS[processor_name]
    out = processor(text, block_size=block_size, font_size=font_size, pad=pad)
    image, mask = out["pixel_values"], out["segment_start_mask"]
    if mirror:
        image = np.ascontiguousarray(image[:, ::-1, :])

    overlay = _draw_patch_boundaries(image, block_size, mask)
    if overlay.shape[1] == 0:
        overlay = np.full((block_size, 1, 3), 255, dtype=np.uint8)
    if upscale > 1:
        overlay = np.repeat(np.repeat(overlay, upscale, axis=0), upscale, axis=1)

    segments = _list_segments(text)
    info = (
        f"segments ({len(segments)}): {segments}\n"
        f"patches: {len(mask)}, segment starts: {int(mask.sum())}"
    )
    return overlay, info


def main():
    with gr.Blocks(title="MPIXEL renderer") as demo:
        gr.Markdown("## MPIXEL rendering processors")
        gr.Markdown(
            "**bigrams**: grapheme pairs within each whitespace-delimited segment. "
            "**adaptive**: cursive scripts → words, CJK → monogram, everything else → bigrams. "
            "**words**: CJK → monogram, everything else → whole-word.\n\n"
            "Red lines mark segment starts (mask == 1). "
            "Gray lines mark the other patch boundaries."
        )
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(
                    label="Text",
                    value="don't wait... Hello, world!",
                    lines=2,
                )
                processor_name = gr.Radio(
                    choices=["bigrams", "adaptive", "words"],
                    value="adaptive",
                    label="processor",
                )
                block_size = gr.Slider(8, 64, value=16, step=1, label="block_size")
                font_size = gr.Slider(6, 32, value=12, step=1, label="font_size")
                with gr.Row():
                    pad_left = gr.Slider(0, 8, value=1, step=1, label="pad_left")
                    pad_right = gr.Slider(0, 8, value=1, step=1, label="pad_right")
                upscale = gr.Slider(1, 12, value=6, step=1, label="display upscale")
                mirror = gr.Checkbox(value=False, label="mirror (horizontal flip)")
            with gr.Column(scale=2):
                out_image = gr.Image(label="Rendered strip", image_mode="RGB")
                out_info = gr.Textbox(label="Info", lines=4, interactive=False)

        inputs = [text, processor_name, block_size, font_size, pad_left, pad_right, upscale, mirror]
        outputs = [out_image, out_info]
        for control in inputs:
            control.change(render, inputs=inputs, outputs=outputs)
        demo.load(render, inputs=inputs, outputs=outputs)

    demo.launch()


if __name__ == "__main__":
    main()
