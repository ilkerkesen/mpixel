"""
Compute recall@K per layer from the .npz files written by run_retrieval.py
and plot one curve per language pair.
"""
import argparse
import glob
import os.path as osp
import re
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "IBM Plex Sans"


def recall_at_k(sim: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-sim, axis=1)[:, :k]
    n = sim.shape[0]
    hits = sum(1 for i in range(n) if i in topk[i])
    return hits / n


def discover_layers(run_name: str, results_dir: str):
    """Return sorted list of (layer_idx, path) from files matching '<run>-layer=<i>.npz'."""
    pattern = osp.join(results_dir, f"{run_name}-layer=*.npz")
    matches = []
    rx = re.compile(rf"^{re.escape(run_name)}-layer=(\d+)\.npz$")
    for path in glob.glob(pattern):
        m = rx.match(osp.basename(path))
        if m:
            matches.append((int(m.group(1)), path))
    matches.sort()
    return matches


def compute_all_recalls(run_name: str, results_dir: str, k: int):
    layers_with_paths = discover_layers(run_name, results_dir)
    if not layers_with_paths:
        raise FileNotFoundError(f"No files matching {run_name}-layer=*.npz in {results_dir}")

    sample = np.load(layers_with_paths[0][1])
    labels = sample["labels"]
    langs = list(dict.fromkeys(labels.tolist()))  # preserve first-occurrence order
    blocks = {lang: np.where(labels == lang)[0] for lang in langs}
    pairs = list(combinations(range(len(langs)), 2))

    layer_ids = [li for li, _ in layers_with_paths]
    rec = np.zeros((len(layer_ids), len(pairs)))

    for row_idx, (_, path) in enumerate(layers_with_paths):
        S = np.load(path)["embeddings"]
        for pair_idx, (i, j) in enumerate(pairs):
            idx_i, idx_j = blocks[langs[i]], blocks[langs[j]]
            # Identify queries/targets that share an index_id so recall@K is well-defined.
            ids_i = sample["indices"][idx_i]
            ids_j = sample["indices"][idx_j]
            common, pos_i, pos_j = np.intersect1d(ids_i, ids_j, return_indices=True)
            if len(common) == 0:
                continue
            sim_ij = S[np.ix_(idx_i[pos_i], idx_j[pos_j])]
            rec[row_idx, pair_idx] = recall_at_k(sim_ij, k=k)

    pair_labels = [f"{langs[i]}–{langs[j]}" for i, j in pairs]
    return layer_ids, rec, pair_labels


def plot_recalls(layers, rec, pair_labels, k: int, save_to: str | None = None):
    plt.figure(figsize=(8, 5))
    for pi, label in enumerate(pair_labels):
        plt.plot(layers, rec[:, pi], marker="o", linestyle="-", label=label)
    plt.xticks(layers)
    plt.xlabel("Layer", fontsize=16)
    plt.ylabel(f"Recall@{k}", fontsize=16)
    plt.legend(loc="best", ncol=2, fontsize=12)
    plt.grid(True, linestyle="--", alpha=1.0)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150)
        print(f"saved {save_to}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-name", required=True,
                   help="Prefix used by run_retrieval.py (filename stem before '-layer=N.npz')")
    p.add_argument("--results-dir", default="outputs/analysis/retrieval")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--save-to", default=None, help="If set, write figure to this path instead of showing it")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    layers, rec, pair_labels = compute_all_recalls(args.run_name, args.results_dir, k=args.K)
    for li, layer in enumerate(layers):
        per_pair = " ".join(f"{lbl}={rec[li, pi]:.3f}" for pi, lbl in enumerate(pair_labels))
        print(f"layer {layer:2d}: {per_pair}")
    plot_recalls(layers, rec, pair_labels, k=args.K, save_to=args.save_to)
