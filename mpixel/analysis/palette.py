"""Shared figure palette for the analysis plots under ``scripts/analysis``.

Every figure in the paper draws its colors from here so a model keeps the same
color across figures. Nothing in this module imports torch (or anything else),
so plot scripts can pull it in cheaply.

Two palettes cover the mpixel family, because the figures ask two different
questions of it:

``MPIXEL_GRADIENT``
    A single-hue ramp, white -> the MPIXEL blue, darkening with the number of
    pretraining languages (1 -> 4 -> 8 -> 8 -> 16). Use it when the *ordering*
    is the point (the curse-of-multilinguality bars), or when the figure also
    shows baselines -- the ramp is entirely blue, so it never collides with the
    orange/gray/black baseline colors.
``MPIXEL_DISTINCT``
    A categorical Okabe-Ito assignment. Use it when several variants are drawn
    as overlapping lines and have to be told apart at a glance (the LINSPECTOR
    layer curves). It reuses hues that ``BASELINE_COLOR`` also uses, so prefer
    ``MPIXEL_GRADIENT`` in figures that mix variants with baselines.

``large-m16`` sits outside both schemes (it is a capacity step, not a language
step) and keeps the same green in either palette.
"""

# Okabe-Ito colorblind-safe palette, by name.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}

# Fallback cycle for series with no fixed color (unknown run names, ad-hoc
# language pairs). Order matches the cycles the plot scripts used before.
OKABE_ITO_CYCLE = [OKABE_ITO[k] for k in
                   ("orange", "blue", "green", "vermillion", "purple", "sky", "yellow", "black")]

# The mpixel family, ordered by pretraining-language count; large-m16 last.
MPIXEL_ORDER = ["base-m1", "base-m4", "base-m8a", "base-m8b", "base-m16", "large-m16"]

# Tints of OKABE_ITO["blue"] mixed with white at 62/42/21/0%. base-m1 is the
# monolingual reference and sits outside the ramp in gray -- as the lightest
# tint it read as a half-transparent bar rather than a model.
MPIXEL_GRADIENT = {
    "base-m1": "#999999",
    "base-m4": "#9EC9E2",
    "base-m8a": "#6BADD2",
    "base-m8b": "#3690C2",
    "base-m16": "#0072B2",
    "large-m16": "#009E73",
}

MPIXEL_DISTINCT = {
    "base-m1": "#999999",
    # Sky, not orange: the LINSPECTOR figure draws base-m4 alongside XLM-R,
    # which owns orange in BASELINE_COLOR.
    "base-m4": "#56B4E9",
    "base-m8a": "#0072B2",
    "base-m8b": "#CC79A7",
    "base-m16": "#D55E00",
    "large-m16": "#009E73",
}

# Non-pixel baselines. Keys cover both spellings the scripts use for XLM-R.
BASELINE_COLOR = {
    "base-xlmr": "#E69F00",
    "xlm-r-base": "#E69F00",
    "xlm-r-large": "#D55E00",
    "canine-s": "#999999",
    "canine": "#999999",
    "mbert": "#000000",
    "bert-base-multilingual-cased": "#000000",
}

MODEL_LABEL = {
    "base-m1": "PIXEL$^{\\dagger}$",
    "base-m4": "MPIXEL$_{4\\mathrm{S}}$",
    "base-m8a": "MPIXEL$_{8\\mathrm{L}}$",
    "base-m8b": "MPIXEL$_{8\\mathrm{S}}$",
    "base-m16": "MPIXEL",
    "large-m16": "MPIXEL$_{\\mathrm{LARGE}}$",
    "base-xlmr": "XLM-R",
    "xlm-r-base": "XLM-R",
    "xlm-r-large": "XLM-R$_\\mathrm{Large}$",
    "canine-s": "CANINE",
    "canine": "CANINE",
    "mbert": "mBERT",
}


def color(model, palette="gradient", index=None):
    """Color for `model`; baselines resolve the same under either palette.

    `palette` is "gradient" or "distinct". Unknown names fall back to
    OKABE_ITO_CYCLE[index] when an `index` is given, else None.
    """
    if model in BASELINE_COLOR:
        return BASELINE_COLOR[model]
    table = MPIXEL_GRADIENT if palette == "gradient" else MPIXEL_DISTINCT
    if model in table:
        return table[model]
    return OKABE_ITO_CYCLE[index % len(OKABE_ITO_CYCLE)] if index is not None else None
