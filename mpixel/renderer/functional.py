from collections import OrderedDict
from functools import lru_cache

import cairo
import gi
import numpy as np
from utf8_tokenizer.control import visualize_control_tokens
from pixel_renderer.renderer import (
    _get_measurement_layout,
    _get_render_surface,
    dim_to_block_size,
    bgra_to_rgb,
    cached_font_description,
    _MAX_RENDER_WIDTH,
)


gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
gi.require_foreign("cairo")
from gi.repository import PangoCairo  # noqa: E402


_UNIT_CACHE_SIZE = 32768
_unit_cache: OrderedDict[tuple, tuple[np.ndarray, int]] = OrderedDict()


@lru_cache(maxsize=_UNIT_CACHE_SIZE)
def _measure_single_unit(
    unit_text: str,
    block_size: int,
    font_size: int,
    pad_left: int,
    pad_right: int,
) -> tuple[int, int, int]:
    layout = _get_measurement_layout()
    layout.set_font_description(cached_font_description("sans", font_size))
    layout.set_text(unit_text, -1)
    w, h = layout.get_pixel_size()
    unit_width = dim_to_block_size(pad_left + w + pad_right, block_size)
    return unit_width, w, h


def _normalize_pad(pad: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(pad, int):
        return pad, 0
    try:
        left, right = pad
    except (TypeError, ValueError) as e:
        raise TypeError(f"pad must be int or length-2 sequence, got {pad!r}") from e
    return int(left), int(right)


def _measure_unit_patches(
    units: list[str],
    block_size: int = 16,
    font_size: int = 12,
    pad: int | tuple[int, int] = 1,
) -> list[int]:
    """Return the number of ``block_size``-wide patches each unit will occupy
    when passed to :func:`_render_units` with the same arguments.

    Useful for building per-patch alignment tensors (e.g. "is this patch the
    start of a new word?") without actually rasterising anything.

    **Reproducibility contract.** This is a low-level helper that resolves
    fonts through whatever fontconfig state the calling process has. It is
    the caller's responsibility to ensure fontconfig is initialised against
    a known font set before calling (otherwise widths depend on host fonts).
    Use :class:`mpixel.renderer.processor.BigramsRenderingProcessor` with
    ``font=FontConfig(...)`` for the reproducible public entry point.
    """
    pad_left, pad_right = _normalize_pad(pad)
    rendered_units = [visualize_control_tokens(u, include_whitespace=True) for u in units]

    layout = _get_measurement_layout()
    layout.set_font_description(cached_font_description("sans", font_size))

    counts = []
    total_width = 0
    for unit in rendered_units:
        layout.set_text(unit, -1)
        w, _ = layout.get_pixel_size()
        new_total = dim_to_block_size(total_width + pad_left + w + pad_right, block_size)
        counts.append((new_total - total_width) // block_size)
        total_width = new_total
    return counts


def _render_units(
    units: list[str],
    block_size: int = 16,
    font_size: int = 12,
    pad: int | tuple[int, int] = 1,
) -> tuple[np.ndarray, list[int]]:
    """Render a sequence of pre-tokenized units onto a single block-aligned strip.

    Each unit is rendered at the next multiple of ``block_size`` — independently
    from its neighbours — so the visual extent of unit ``i`` never bleeds into
    unit ``i + 1``. Useful when downstream processing needs a known
    patch-to-unit mapping (e.g. one unit per word, or per n-gram).

    Args:
        units: Text fragments to render in order. Callers are responsible for
            whatever tokenization they want (``str.split()``, grapheme n-grams,
            language-aware segmentation, etc.).
        block_size: Height of the output in pixels; each unit's width is also
            rounded up to a multiple of this.
        font_size: Font size.
        pad: Horizontal padding in pixels around each unit. An ``int`` pads both
            sides equally; a ``(left, right)`` tuple sets them independently.

    Returns:
        ``(image, patch_counts)``. ``image`` is the rendered RGB array of shape
        ``(block_size, total_width, 3)`` (zero-width when ``units`` is empty).
        ``patch_counts[i]`` is the number of ``block_size``-wide patches unit
        ``i`` occupies in ``image``, suitable for building per-patch alignment
        tensors without a separate measurement pass.

    Note:
        Like :func:`_measure_unit_patches`, this is low-level plumbing: font
        resolution relies on the current process's fontconfig state. For
        reproducible rendering use
        :class:`mpixel.renderer.processor.BigramsRenderingProcessor` with an
        explicit ``font=FontConfig(...)``.
    """
    pad_left, pad_right = _normalize_pad(pad)
    cache_key_base = (block_size, font_size, pad_left, pad_right)

    rendered_units = [visualize_control_tokens(u, include_whitespace=True) for u in units]

    # Check cache for each unit; collect misses
    results: list[tuple[np.ndarray, int] | None] = []
    miss_indices: list[int] = []
    for i, unit in enumerate(rendered_units):
        key = (unit,) + cache_key_base
        if key in _unit_cache:
            _unit_cache.move_to_end(key)
            results.append(_unit_cache[key])
        else:
            results.append(None)
            miss_indices.append(i)

    # Batch render all misses on a single surface
    if miss_indices:
        layout = _get_measurement_layout()
        layout.set_font_description(cached_font_description("sans", font_size))

        miss_measured: list[tuple[int, int, int, int]] = []
        miss_total_width = 0
        for i in miss_indices:
            unit_width, w, h = _measure_single_unit(
                rendered_units[i], block_size, font_size, pad_left, pad_right
            )
            miss_measured.append((miss_total_width, w, h, unit_width))
            miss_total_width += unit_width

        if miss_total_width > 0:
            if miss_total_width <= _MAX_RENDER_WIDTH:
                surface, context = _get_render_surface(block_size)
                surface_width = _MAX_RENDER_WIDTH
            else:
                surface = cairo.ImageSurface(cairo.FORMAT_RGB24, miss_total_width, block_size)
                context = cairo.Context(surface)
                surface_width = miss_total_width

            context.set_source_rgb(1.0, 1.0, 1.0)
            context.rectangle(0, 0, miss_total_width, block_size)
            context.fill()
            context.set_source_rgb(0.0, 0.0, 0.0)

            for i, (offset, w, h, unit_width) in zip(miss_indices, miss_measured):
                layout.set_text(rendered_units[i], -1)
                context.move_to(offset + pad_left, (block_size - h) // 2)
                PangoCairo.show_layout(context, layout)

            data = surface.get_data()
            bgra = np.frombuffer(data, dtype=np.uint8).reshape(
                (block_size, surface_width, 4)
            )
            if surface_width > miss_total_width:
                bgra = bgra[:, :miss_total_width, :]
            miss_rgb = bgra_to_rgb(bgra)

            for i, (offset, w, h, unit_width) in zip(miss_indices, miss_measured):
                strip = miss_rgb[:, offset:offset + unit_width, :].copy()
                strip.flags.writeable = False
                patch_count = unit_width // block_size
                key = (rendered_units[i],) + cache_key_base
                if len(_unit_cache) >= _UNIT_CACHE_SIZE:
                    _unit_cache.popitem(last=False)
                _unit_cache[key] = (strip, patch_count)
                results[i] = (strip, patch_count)
        else:
            for i in miss_indices:
                strip = np.full((block_size, 0, 3), 255, dtype=np.uint8)
                strip.flags.writeable = False
                key = (rendered_units[i],) + cache_key_base
                if len(_unit_cache) >= _UNIT_CACHE_SIZE:
                    _unit_cache.popitem(last=False)
                _unit_cache[key] = (strip, 0)
                results[i] = (strip, 0)

    # Assemble final strip
    strips = []
    patch_counts = []
    for strip, count in results:
        strips.append(strip)
        patch_counts.append(count)

    if not strips:
        return np.full((block_size, 0, 3), 255, dtype=np.uint8), patch_counts

    image = np.concatenate(strips, axis=1)
    return image, patch_counts
