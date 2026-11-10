#!/usr/bin/env python3
"""Curse-of-multilinguality bar plot for the mpixel models across benchmarks.

Two panels share a y-axis. Groups are benchmarks; within each group the bars are
the mpixel models ordered by pretraining-language count -- base-m4 (4) ->
base-m8a (8) -> base-m8b (8) -> base-m16 (16) -> large-m16 (16, larger). Each
bar is one score per (model, benchmark): the mean over a language set (no
per-language bars).

  * left  -- Pretraining languages: the 4 core languages base-m4 was pretrained
             on (eng, hin, zho, ukr), which every mpixel model shares (m4's set
             is nested in m8a/m8b/m16). As languages are added, capacity per
             language shrinks and performance on these shared languages drops --
             the curse of multilinguality.
  * right -- Non-pretraining languages: every language base-m16 is evaluated on
             that is NOT in the m16 pretraining set. GLUE is English-only, so it
             has no non-pretraining languages and is dropped from this panel.

Each benchmark averages over whatever subset of the language set it evaluates
(e.g. NER and XNLI have no Ukrainian, so their pretraining-core bar is a
3-language mean); the language count is annotated under each benchmark label.
Point estimates reuse the loaders in scripts/utils/summary_table.py.
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mpixel.analysis.palette import MODEL_LABEL
from mpixel.analysis.palette import MPIXEL_GRADIENT as MODEL_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import summary_table as S  # noqa: E402

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "IBM Plex Sans"
# The model labels are mathtext (MPIXEL$_{16}$), so point mathtext at the same
# family -- otherwise the subscripts fall back to DejaVu next to Plex text.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "IBM Plex Sans"
plt.rcParams["mathtext.it"] = "IBM Plex Sans:italic"
plt.rcParams["mathtext.bf"] = "IBM Plex Sans:bold"
plt.rcParams["mathtext.cal"] = "IBM Plex Sans:italic"  # else falls back to 'cursive'

# Models, ordered by pretraining-language count (the curse axis): PIXEL is the
# monolingual (English-only) reference; the mpixel models add languages 4->16.
# Labels and the language-count gradient come from mpixel.analysis.palette
# (imported above) so they match the other figures.
MODELS = ["base-m1", "base-m4", "base-m8a", "base-m8b", "base-m16", "large-m16"]

# base-m4's 4 pretraining languages, shared by every mpixel model.
CORE_LANGS = ["eng", "hin", "zho", "ukr"]

# Reference model whose eval languages define each benchmark's language set.
REF_MODEL = "base-m16"

# Y-axis limits. The bottom clips the bars rather than drawing them from 0; the
# top stops just past the tallest bar (89.6) so the last tick is 90, not 100.
Y_MIN, Y_MAX = 50, 92


def load_benchmark(bench, log_dir, N):
    if bench.get("loader") == "xnli":
        point, _ = S.load_xnli(log_dir, N, False)
    else:
        point, _ = S.load_points(log_dir, bench["select"], bench["report"], N, False, bench["keyinfo"])
    return point


def lang_keys_of(point, keyinfo):
    """language -> [eval keys] for REF_MODEL (GLUE: one language 'eng' -> all tasks)."""
    lk = defaultdict(list)
    for (model, key) in point:
        if model != REF_MODEL:
            continue
        info = keyinfo(key)
        if info:
            lk[info[0]].append(key)
    return lk


def score(point, model, langs, lang_keys):
    """Mean over `langs` (each language = mean over its keys), present only."""
    per_lang = []
    for lang in langs:
        vals = [point[(model, k)] for k in lang_keys[lang] if (model, k) in point]
        if vals:
            per_lang.append(float(np.mean(vals)))
    return float(np.mean(per_lang)) if per_lang else None


def build(dirs, N):
    """Return {'pt': [...], 'nonpt': [...]} of (display, count, {model: score})."""
    panels = {"pt": [], "nonpt": []}
    for b in S.BENCHMARKS:
        point = load_benchmark(b, dirs[b["name"]], N)
        if not point:
            print(f"[{b['name']}: no runs under {dirs[b['name']]}, skipped]", file=sys.stderr)
            continue
        lk = lang_keys_of(point, b["keyinfo"])
        core = [l for l in CORE_LANGS if l in lk]
        nonpt = sorted(l for l in lk if l not in S.PRETRAIN_LANGS)
        if core:
            panels["pt"].append((b["display"], len(core),
                                 {m: score(point, m, core, lk) for m in MODELS}))
        if nonpt:
            panels["nonpt"].append((b["display"], len(nonpt),
                                    {m: score(point, m, nonpt, lk) for m in MODELS}))
        print(f"[{b['name']}: pt-core={len(core)} non-pt={len(nonpt)}]", file=sys.stderr)
    return panels


def plot_panel(ax, groups, title):
    x = np.arange(len(groups))
    n = len(MODELS)
    width = 0.82 / n
    for i, model in enumerate(MODELS):
        heights = [g[2].get(model) or 0.0 for g in groups]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=MODEL_LABEL[model],
               color=MODEL_COLOR[model], zorder=3, linewidth=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n($n={c}$)" for d, c, _ in groups], fontsize=19)
    if title:
        ax.set_title(title, fontsize=21)
    ax.tick_params(axis="y", labelsize=19)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.9)


# Leaves a band at the bottom of the figure for the legend.
TIGHT_RECT = (0, 0.07, 1, 1)


def widen_to_legend(fig, leg, tol=0.02, max_passes=4):
    """Grow the figure until the axes are as wide as the one-row legend.

    The legend is a fixed width (six keys of fixed text), so a figure narrow
    enough to make it overhang leaves the bars squeezed under a much wider key.
    Widening the figure by the deficit leaves the margins alone, so the axes
    pick up the whole increase; two passes are normally enough to converge.
    """
    for _ in range(max_passes):
        fig.canvas.draw()
        axes = fig.get_axes()
        span = (max(a.get_window_extent().x1 for a in axes)
                - min(a.get_window_extent().x0 for a in axes))
        deficit = (leg.get_window_extent().width - span) / fig.dpi
        if deficit <= tol:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + deficit, h)
        fig.tight_layout(rect=TIGHT_RECT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sib", default="/scratch1/ilker/sib-results")
    parser.add_argument("--ner", default="/scratch1/ilker/ner-results")
    parser.add_argument("--udp", default="/scratch1/ilker/udp-results")
    parser.add_argument("--xnli", default="/scratch1/ilker/xnli-results")
    parser.add_argument("--glue", default="/scratch1/ilker/glue-results")
    parser.add_argument("--N", type=int, default=3, help="Average the best N runs.")
    parser.add_argument("--panels", choices=["both", "pt", "nonpt"], default="both",
                        help="Which panel(s) to draw: both (default), pt (pretraining "
                             "languages only), or nonpt (non-pretraining languages only).")
    parser.add_argument("--out", default="outputs/analysis/curse_of_multilinguality",
                        help="Output path prefix; .pdf and .png are written.")
    args = parser.parse_args()

    dirs = {"sib": args.sib, "ner": args.ner, "udp": args.udp,
            "xnli": args.xnli, "glue": args.glue}
    panels = build(dirs, args.N)

    if args.panels == "both":
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        plot_panel(axL, panels["pt"], "Pretraining languages")
        plot_panel(axR, panels["nonpt"], "Non-pretraining languages")
    else:
        fig, axL = plt.subplots(1, 1, figsize=(6.8, 6.3))
        # Single panel -> no title; which language set it is belongs in the caption.
        plot_panel(axL, panels[args.panels], None)
    axL.set_ylabel("Performance", fontsize=21)
    # Truncated y-axis: every bar is well above 50, so starting there spends the
    # same figure height on half the range and makes the gaps between models
    # readable. The figure size is unchanged.
    axL.set_ylim(Y_MIN, Y_MAX)

    handles, labels = axL.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), fontsize=18,
                     frameon=True, framealpha=0.9, handlelength=1.2, handletextpad=0.5,
                     columnspacing=1.2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=TIGHT_RECT)
    widen_to_legend(fig, leg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
