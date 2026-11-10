"""Rank (checkpoint, layer) pairs by average recall@K from one validation JSON.

Reads a JSON produced by run_flores.py / run_tatoeba.py, computes the cross-pair
average recall@K at each (step, layer), and prints the top --top entries ranked
by score. A single checkpoint can appear multiple times in the ranking — once
per layer.
"""
import argparse
import json
import os.path as osp


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_entries(
    data: dict, k: int, period: int | None, languages: set[str] | None,
):
    """Yield (step, layer_key, avg_recall@k) for every (step, layer) in the JSON."""
    # JSON object keys are strings; cast step keys back to int for sorting.
    step_items = sorted(
        ((int(s), c) for s, c in data["checkpoints"].items()),
        key=lambda x: x[0],
    )
    if period is not None:
        step_items = [(s, c) for s, c in step_items if s % period == 0]

    metric_key = f"recall@{k}"
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

        layer_keys = sorted(ckpt["pairs"][pair_keys[0]].keys(), key=int)
        for lk in layer_keys:
            avg = sum(ckpt["pairs"][pk][lk][metric_key] for pk in pair_keys) / len(pair_keys)
            yield step, lk, avg


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Path to validation JSON file")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--top", type=int, default=10, help="Number of top entries to print")
    p.add_argument("--period", type=int, default=None,
                   help="Only include checkpoints whose step is a multiple of this value")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Restrict the average to pairs whose both endpoints are in this "
                        "set; defaults to every language present")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not osp.isfile(args.input):
        raise FileNotFoundError(args.input)

    languages = set(args.languages) if args.languages else None
    data = load_json(args.input)

    entries = list(collect_entries(data, args.K, args.period, languages))
    if not entries:
        print("No entries found.")
        raise SystemExit(0)

    entries.sort(key=lambda x: -x[2])
    top = entries[:args.top]

    print(f"{'rank':>4}  {'step':>10}  {'layer':>5}  recall@{args.K}")
    for rank, (step, layer, avg) in enumerate(top, 1):
        print(f"{rank:>4}  {step:>10}  {layer:>5}  {avg:.4f}")
