"""Plot LINSPECTOR probing results written by run_linspector.py.

Reads ``<results-dir>/<model>-<lang>-linspector.json`` files and draws
test-accuracy-vs-relative-depth curves (one line per model) in one of two
layouts:

``grid`` (default)
    A single figure, one row per task and one column per language.
``per-task``
    One figure per task, one panel per language wrapped into at most
    ``--max-cols`` columns. ``--save-to`` is then a directory receiving
    ``<task>.<ext>``, and ``--tex-out`` writes a ``figure*`` environment for
    every task.

Languages are always drawn in alphabetical order of their display name.
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mpixel.analysis.palette import MODEL_LABEL, color as model_color


MARKERS = ['o', 's', '^', 'D', 'v']
MAIN_FONT_SCALE = 1.5
SINGLE_COLUMN_TASKS = {"Comparison", "Polarity"}
TASKS_AT_END = ("Comparison", "Polarity")

LANG_TITLE = {
    "modern-greek": "Greek",
    "serbian_latin": "Serbian (Latin)",
    "serbo-croatian": "Serbo-Croatian",
}

# Shared by every per-task figure; only the task name changes.
CAPTION = (
    "\\textsc{{linspector}} probing results for the \\textit{{{task}}} task. "
    "Each panel shows linear probe accuracy per layer, with the x-axis giving "
    "the layer position relative to the model depth."
)


def load_run(results_dir, model, lang):
    path = osp.join(results_dir, f"{model}-{lang}-linspector.json")
    if not osp.isfile(path):
        print(f"Warning: {path} not found.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def lang_title(lang):
    return LANG_TITLE.get(lang, lang.capitalize())


def sort_languages(languages):
    return sorted(languages, key=lang_title)


def find_languages(results_dir, models):
    suffix = "-linspector.json"
    languages = set()
    for model in models:
        for path in glob.glob(osp.join(results_dir, f"{model}-*{suffix}")):
            languages.add(osp.basename(path)[len(model) + 1:-len(suffix)])
    return languages


def setup_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'IBM Plex Sans'
    # The model labels are mathtext (MPIXEL$_{\mathrm{LARGE}}$), so point
    # mathtext at the same family or the subscript falls back to DejaVu.
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'IBM Plex Sans'
    plt.rcParams['mathtext.it'] = 'IBM Plex Sans:italic'
    plt.rcParams['mathtext.bf'] = 'IBM Plex Sans:bold'


def plot_curves(ax, runs, args, task, lang):
    for m, model in enumerate(args.models):
        data = runs.get((model, lang))
        if data is None or task not in data.get("results", {}):
            continue

        task_data = data["results"][task]
        layers = sorted(int(k) for k in task_data)
        test_accs = [task_data[str(layer)]["test_acc"] for layer in layers]
        # Relative depth, so 12- and 24-layer models are comparable.
        depth = [100.0 * layer / layers[-1] for layer in layers]
        ax.plot(
            depth,
            test_accs,
            label=(args.model_labels[m] if args.model_labels
                   else MODEL_LABEL.get(model, model)),
            marker=MARKERS[m % len(MARKERS)],
            linestyle='-',
            # Mixes mpixel variants with an XLM-R baseline, so the
            # gradient palette (all blue, never collides with the
            # baseline hues); unknown run names fall back to the cycle.
            color=model_color(model, "gradient", index=m),
        )

    ax.set_xlim(0, 102)
    ax.xaxis.set_major_locator(MultipleLocator(25))


def finish_figure(fig, handles, labels, save_to, font_scale=1.0):
    # Bottom band, top to bottom: panels, x-label, legend. Sized from the
    # legend's measured height rather than a guessed fraction, so the band is
    # exactly what the two pieces need and no empty strip is left over.
    fig_h = fig.get_size_inches()[1]
    pad = 0.04 / fig_h                          # ~0.04in, in figure fraction
    legend_top = 0.0
    if handles:
        # One legend row when it fits the figure width, else fewer columns.
        for ncol in range(len(labels), 0, -1):
            leg = fig.legend(handles, labels, loc='lower center', ncol=ncol,
                             fontsize=16 * font_scale, frameon=True, framealpha=0.9,
                             handlelength=1.6, handletextpad=0.6, columnspacing=1.8,
                             bbox_to_anchor=(0.5, 0.0))
            fig.canvas.draw()
            extent = leg.get_window_extent().transformed(fig.transFigure.inverted())
            if extent.width <= 1.0 or ncol == 1:
                break
            leg.remove()
        legend_top = extent.y1
    fig.supxlabel('Layer Proportional (%)', fontsize=20 * font_scale,
                  y=legend_top + pad, va='bottom')
    # rect stops just above the legend; tight_layout reserves the supxlabel's
    # own height on top of that, so adding it here would double-count it.
    fig.tight_layout(rect=(0, legend_top + pad, 1, 1), pad=0.5)

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"saved {save_to}")
    else:
        plt.show()
    plt.close(fig)


def plot_grid(args):
    languages = sort_languages(args.languages)
    tasks = args.tasks

    runs = {(model, lang): load_run(args.results_dir, model, lang)
            for model in args.models for lang in languages}

    fig, axes = plt.subplots(
        nrows=len(tasks),
        ncols=len(languages),
        figsize=(4 * len(languages), 3 * len(tasks)),
        sharey="row",
        sharex="col",
        squeeze=False,
    )

    for i, task in enumerate(tasks):
        for j, lang in enumerate(languages):
            ax = axes[i, j]
            plot_curves(ax, runs, args, task, lang)
            ax.tick_params(axis='both', labelsize=10 * MAIN_FONT_SCALE)
            if i == 0:
                ax.set_title(lang_title(lang), pad=10, fontsize=24 * MAIN_FONT_SCALE)
            if j == 0:
                ax.set_ylabel(task, rotation=90, labelpad=20, va='center',
                              fontsize=20 * MAIN_FONT_SCALE)

    handles, labels = axes[-1, 0].get_legend_handles_labels()
    finish_figure(fig, handles, labels, args.save_to, font_scale=MAIN_FONT_SCALE)


def plot_per_task(args):
    languages = args.languages or find_languages(args.results_dir, args.models)
    runs = {}
    for model in args.models:
        for lang in languages:
            data = load_run(args.results_dir, model, lang)
            if data is not None:
                runs[(model, lang)] = data

    all_tasks = sorted({task for data in runs.values() for task in data["results"]})
    tasks = args.tasks or all_tasks
    tasks = [task for task in tasks if task not in TASKS_AT_END] + [
        task for task in TASKS_AT_END if task in tasks
    ]
    if args.save_to:
        os.makedirs(args.save_to, exist_ok=True)

    ncols_per_task = {}
    for task in tasks:
        task_langs = sort_languages(
            {lang for (_, lang), data in runs.items() if task in data["results"]})
        if not task_langs:
            print(f"Warning: no results for task {task}, skipping.")
            continue

        ncols = 1 if task in SINGLE_COLUMN_TASKS else min(args.max_cols, len(task_langs))
        nrows = math.ceil(len(task_langs) / ncols)
        ncols_per_task[task] = ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for k, ax in enumerate(axes.flat):
            if k >= len(task_langs):
                ax.set_visible(False)
                continue
            lang = task_langs[k]
            plot_curves(ax, runs, args, task, lang)
            ax.set_title(lang_title(lang), pad=10, fontsize=24)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        save_to = osp.join(args.save_to, f"{task}.{args.ext}") if args.save_to else None
        finish_figure(fig, handles, labels, save_to)

    if args.tex_out:
        write_tex(args.tex_out, ncols_per_task, args)


def write_tex(path, ncols_per_task, args):
    blocks = []
    for task, ncols in ncols_per_task.items():
        # Narrow figures keep the per-panel scale of the full 5-column ones.
        width = "\\textwidth" if ncols == args.max_cols \
            else f"{ncols / args.max_cols:.2f}\\textwidth"
        blocks.append(
            "\\begin{figure*}[!htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width={width}]{{{args.tex_fig_dir}/{task}.{args.ext}}}\n"
            f"  \\caption{{{CAPTION.format(task=task)}}}\n"
            f"  \\label{{fig:linspector-{task.lower()}}}\n"
            "\\end{figure*}\n"
        )
    with open(path, "w") as f:
        f.write("\n".join(blocks))
    print(f"saved {path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Run names (filename stems before '-<lang>-linspector.json')")
    p.add_argument("--pattern", default=None,
                   help="Only plot models whose run name matches this regex pattern")
    p.add_argument("--model-labels", nargs="+", default=None,
                   help="Legend labels for each model (defaults to the run names)")
    p.add_argument("--layout", choices=["grid", "per-task"], default="grid")
    p.add_argument("--languages", nargs="+", default=None,
                   help="grid default: arabic modern-greek russian spanish turkish; "
                        "per-task default: every language with results")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="grid default: Person TagCount; per-task default: every task with results")
    p.add_argument("--max-cols", type=int, default=4,
                   help="per-task: maximum number of language columns per figure")
    p.add_argument("--results-dir", default="outputs/analysis/linspector")
    p.add_argument("--save-to", default=None,
                   help="If set, write the figure here instead of showing it "
                        "(per-task: a directory receiving <task>.<ext>)")
    p.add_argument("--ext", default="pdf", help="per-task: figure file extension")
    p.add_argument("--tex-out", default=None,
                   help="per-task: write the LaTeX figure environments to this file")
    p.add_argument("--tex-fig-dir", default="figures/linspector",
                   help="per-task: figure directory as referenced from the LaTeX project")
    args = p.parse_args()
    if args.model_labels and len(args.model_labels) != len(args.models):
        p.error("--model-labels must have the same number of entries as --models")

    if args.layout == "grid":
        args.languages = args.languages or ["arabic", "modern-greek", "russian",
                                            "spanish", "turkish"]
        args.tasks = args.tasks or ["Person", "TagCount"]

    # Keep only models matching the regex, preserving model<->label alignment.
    if args.pattern is not None:
        regex = re.compile(args.pattern)
        pairs = list(zip(args.models, args.model_labels)) if args.model_labels \
            else [(m, None) for m in args.models]
        kept = [(m, lbl) for m, lbl in pairs if regex.search(m)]
        if not kept:
            p.error(f"No --models match pattern '{args.pattern}': {args.models}")
        args.models = [m for m, _ in kept]
        args.model_labels = [lbl for _, lbl in kept] if args.model_labels else None

    args.results_dir = osp.abspath(osp.expanduser(args.results_dir))
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_style()
    if args.layout == "grid":
        plot_grid(args)
    else:
        plot_per_task(args)
