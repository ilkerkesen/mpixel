#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--metric", default="val_acc")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--lower-is-better", action="store_true")
    args = parser.parse_args()

    runs = [json.load(open(p)) for p in Path(args.log_dir).rglob("all_results.json")]
    runs = [r for r in runs if args.metric in r]
    runs.sort(key=lambda x: x[args.metric], reverse=not args.lower_is_better)
    best = runs[: args.N]

    output = {
        k: {"mean": round(np.mean(v) * 100, 2), "std": round(np.std(v) * 100, 2)}
        for k in best[0] if "loss" not in k for v in [[r[k] for r in best if k in r]] if v
    }
    lines = ["{"] + [f'  "{k}": {json.dumps(v)},' for k, v in output.items()]
    lines[-1] = lines[-1].rstrip(",")
    print("\n".join(lines) + "\n}")

if __name__ == "__main__":
    main()
