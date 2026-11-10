"""Plot average recall@K across training iterations from validation JSONs.

For each JSON in --input-dir (produced by run_flores.py / run_tatoeba.py),
plots recall@K averaged across language pairs at the model's last layer
against iteration. One curve per JSON. Same style as plot_retrieval.py.
"""
import argparse
import glob
import json
import os.path as osp

import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "IBM Plex Sans"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_avg_curve(
    data: dict, k: int, period: int | None, languages: set[str] | None,
    layer: int | str | None,
):
    """Return (steps, [avg_recall@k per step], [layer_key per step]) for one JSON.

    Layer selection per step:
      - ``layer`` is None: use the highest-index layer.
      - ``layer`` is an int: use that layer.
      - ``layer`` == "best": per-step pick the layer that maximizes the
        cross-pair average recall@k.

    If ``languages`` is given, only pairs whose both endpoints are in the set
    contribute to the average; otherwise all pairs are used.
    """
    # JSON object keys are strings; cast step keys back to int for sorting.
    step_items = sorted(
        ((int(s), c) for s, c in data["checkpoints"].items()),
        key=lambda x: x[0],
    )
    if period is not None:
        step_items = [(s, c) for s, c in step_items if s % period == 0]

    if not step_items:
        return [], [], []

    metric_key = f"recall@{k}"
    steps, avgs, layers_used = [], [], []
    for step, ckpt in step_items:
        pair_keys = list(ckpt["pairs"].keys())
        if languages is not None:
            # pair_key is built as f"{lang1}-{lang2}" and language codes here
            # never contain '-', so a single split is unambiguous.
            pair_keys = [
                pk for pk in pair_keys
                if set(pk.split("-", 1)) <= languages
            ]
        if not pair_keys:
            continue

        # All pairs share the same layer set (one forward pass per checkpoint).
        layer_keys = sorted(ckpt["pairs"][pair_keys[0]].keys(), key=int)

        def avg_at(lk: str) -> float:
            return sum(ckpt["pairs"][pk][lk][metric_key] for pk in pair_keys) / len(pair_keys)

        if layer == "best":
            chosen_lk = max(layer_keys, key=avg_at)
        elif layer is None:
            chosen_lk = layer_keys[-1]
        else:
            chosen_lk = str(layer)

        steps.append(step)
        avgs.append(avg_at(chosen_lk))
        layers_used.append(chosen_lk)
    return steps, avgs, layers_used


def _layer_arg(s: str):
    if s == "best":
        return "best"
    return int(s)


def format_step(s: int) -> str:
    if s == 0:
        return "0"
    if s >= 1_000_000:
        return f"{s/1_000_000:g}M"
    return f"{s/1000:g}k"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory containing validation JSON files")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--period", type=int, default=None,
                   help="Only include checkpoints whose step is a multiple of this value")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Restrict the average to pairs whose both endpoints are in this "
                        "set; defaults to every language present in each JSON")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--layer", type=_layer_arg, default=None,
                   help="Layer index, or 'best' to pick the per-checkpoint optimum; "
                        "defaults to the last (highest) layer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_paths = sorted(glob.glob(osp.join(args.input_dir, "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files in {args.input_dir}")

    languages = set(args.languages) if args.languages else None

    plt.figure(figsize=(8, 5))
    all_steps: set[int] = set()
    for path in json_paths:
        data = load_json(path)
        steps, avgs, layers_used = extract_avg_curve(
            data, args.K, args.period, languages, args.layer,
        )
        if not steps:
            print(f"skipped {osp.basename(path)} (no matching checkpoints/pairs)")
            continue

        stem = osp.splitext(osp.basename(path))[0]
        print(f"\n=== {stem} ===")
        for step, avg, lk in zip(steps, avgs, layers_used):
            print(f"step {step:>8d}: recall@{args.K}={avg:.3f} (layer {lk})")

        all_steps.update(steps)
        plt.plot(steps, avgs, marker="o", linestyle="-", label=stem)

    if all_steps:
        tick_period = 100_000
        lo, hi = min(all_steps), max(all_steps)
        start = ((lo + tick_period - 1) // tick_period) * tick_period
        end = (hi // tick_period) * tick_period
        if start <= end:
            ticks = list(range(start, end + 1, tick_period))
            plt.xticks(ticks, [format_step(s) for s in ticks])
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel(f"Avg Recall@{args.K}", fontsize=16)
    if args.title:
        plt.title(args.title, fontsize=14)
    plt.legend(loc="best", ncol=2, fontsize=12)
    plt.grid(True, linestyle="--", alpha=1.0)
    plt.tight_layout()
    plt.show()
