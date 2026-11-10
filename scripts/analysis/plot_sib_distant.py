#!/usr/bin/env python3
"""Per-script SIB-200 bar plot for a curated set of typologically distant languages.

Reads the same on-disk layout as scripts/utils/sib_table.py
(log-dir/model/lang/lr--seed/all_results.json, runs ranked by --metric and the
best --N averaged on --report) plus an optional XLM-R baseline directory
(<lang>_xlmrbase/test_result_*.txt with an "f1 = ..." line). The language set
comes from a CSV with `code` and `script` columns (e.g. distant_languages.csv);
per-script bars average over all CSV languages of that script that a model has
a result for.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mpixel.analysis.palette import BASELINE_COLOR, MPIXEL_GRADIENT

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'IBM Plex Sans'

MODELS = ["base-xlmr", "canine-s", "base-m16"]
# Short labels here (one mpixel model, so no subscript is needed); colors come
# from mpixel.analysis.palette -- the gradient palette for the mpixel model,
# since this figure mixes it with baselines.
MODEL_LABEL = {
    "base-xlmr": "XLM-R",
    "canine-s": "CANINE",
    "base-m16": "MPIXEL",
}
MODEL_COLOR = {
    "base-xlmr": BASELINE_COLOR["base-xlmr"],
    "canine-s": BASELINE_COLOR["canine-s"],
    "base-m16": MPIXEL_GRADIENT["base-m16"],
}

SCRIPT_ORDER = lambda s: (s != "Latn", s)  # Latin first, then alphabetical


def script_of(lang):
    return lang.split("_", 1)[1] if "_" in lang else ""


def load_distant_langs(csv_path):
    """code_Script entries, in CSV row order (dedup, since taq appears twice)."""
    langs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            lang = f"{row['code']}_{row['script']}"
            if lang not in langs:
                langs.append(lang)
    return langs


def load_points(log_dir, xlm_r_output_dir, langs, metric, report, N):
    """(model, lang) -> averaged report metric (best N runs by metric), *100."""
    point = {}
    for model in MODELS:
        if model == "base-xlmr":
            continue
        model_dir = Path(log_dir) / model
        if not model_dir.is_dir():
            continue
        for lang in langs:
            lang_dir = model_dir / lang
            if not lang_dir.is_dir():
                continue
            runs = []
            for rf in lang_dir.rglob("all_results.json"):
                with open(rf) as f:
                    runs.append(json.load(f))
            runs = [r for r in runs if metric in r]
            if not runs:
                continue
            runs.sort(key=lambda r: r[metric], reverse=True)
            vals = [r[report] for r in runs[:N] if report in r]
            if vals:
                point[(model, lang)] = float(np.mean(vals)) * 100

    if xlm_r_output_dir:
        langs_set = set(langs)
        xlmr_large = {}
        for d in sorted(Path(xlm_r_output_dir).iterdir()):
            if not d.is_dir():
                continue
            lang, _, variant = d.name.rpartition("_")
            if lang not in langs_set or variant not in ("xlmrbase", "xlmr"):
                continue
            f1s = []
            for fp in sorted(d.glob("test_result_*.txt")):
                with open(fp) as f:
                    for line in f:
                        key, sep, val = line.partition("=")
                        if sep and key.strip() == "f1":
                            try:
                                f1s.append(float(val))
                            except ValueError:
                                pass
                            break
            if not f1s:
                continue
            mean_f1 = float(np.mean(f1s)) * 100
            if variant == "xlmrbase":
                point[("base-xlmr", lang)] = mean_f1
            else:
                xlmr_large[lang] = mean_f1
        # Fall back to large XLM-R where base has no result for a wanted lang.
        for lang, v in xlmr_large.items():
            point.setdefault(("base-xlmr", lang), v)

    return point


def per_script_means(point, langs):
    """script -> {model: mean over langs of that script present for model}."""
    by_script = defaultdict(list)
    for lang in langs:
        by_script[script_of(lang)].append(lang)

    means = {}
    for script, script_langs in by_script.items():
        means[script] = {}
        for model in MODELS:
            vs = [point[(model, l)] for l in script_langs if (model, l) in point]
            if vs:
                means[script][model] = float(np.mean(vs))
    return means


def plot(means, out=None):
    scripts = sorted(means, key=SCRIPT_ORDER)
    x = np.arange(len(scripts))
    n_models = len(MODELS)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, model in enumerate(MODELS):
        heights = [means[s].get(model, 0.0) for s in scripts]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=MODEL_LABEL[model],
               color=MODEL_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(scripts, fontsize=12, rotation=45, ha="center", rotation_mode="anchor")
    ax.tick_params(axis='x', pad=14)
    ax.set_ylabel("Performance", fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(loc="upper left", ncol=n_models, fontsize=12, frameon=True, framealpha=0.9,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.0, borderpad=0.4)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle='--', alpha=1.0, linewidth=0.5)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distant-csv",
                        default="/scratch1/ilker/distals-outputs-v2/distant_languages.csv")
    parser.add_argument("--log-dir", default="/scratch1/ilker/sib-results")
    parser.add_argument("--xlm-r-output-dir", default="/scratch1/ilker/xlm-roberta-outputs")
    parser.add_argument("--metric", default="val_f1", help="Selection metric.")
    parser.add_argument("--report", default="test_f1", help="Reported metric.")
    parser.add_argument("--N", type=int, default=3, help="Average the best N runs.")
    parser.add_argument("--out", help="Optional path to also save the figure to.")
    args = parser.parse_args()

    langs = load_distant_langs(args.distant_csv)
    point = load_points(args.log_dir, args.xlm_r_output_dir, langs, args.metric, args.report, args.N)
    means = per_script_means(point, langs)
    plot(means, args.out)


if __name__ == "__main__":
    main()
