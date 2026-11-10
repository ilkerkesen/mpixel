"""Bitext retrieval evaluation on FLORES (offline, from pre-downloaded per-language data).

Follows the same methodology as run_bitext_mining.py: embed all languages into
one matrix, center globally, L2-normalize, then compute pairwise recall from
the single pre-processed matrix.
"""
import argparse
import itertools
import json
import os
import os.path as osp
import re
import sys
from typing import Dict

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS

from mpixel.datamodules.sib200 import _make_renderer, preprocess_text_for_classification
from mpixel.utils import get_data_dir
from mpixel.utils.checkpoint import load_model_from_checkpoint
from mpixel.utils.embedding import embed_batch
from mpixel.utils.preprocessing import get_transforms


TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
RECALL_KS = [1, 5, 10]
_STEP_RE = re.compile(r"step=(\d+)")


def parse_step(filename: str) -> int | None:
    m = _STEP_RE.search(filename)
    return int(m.group(1)) if m else None


def collect_checkpoints(
    paths: list[str], period: int | None,
) -> list[tuple[int, str]]:
    ckpts = []
    dirs = []
    for p in paths:
        p = osp.abspath(osp.expanduser(p))
        if osp.isfile(p) and p.endswith(".ckpt"):
            step = parse_step(osp.basename(p))
            ckpts.append((step if step is not None else 0, p))
        elif osp.isdir(p):
            dirs.append(p)
            for f in sorted(os.listdir(p)):
                if f.endswith(".ckpt"):
                    step = parse_step(f)
                    if step is not None:
                        ckpts.append((step, osp.join(p, f)))
    if period is not None:
        ckpts = [(s, p) for s, p in ckpts if s % period == 0]
    ckpts.sort(key=lambda x: x[0])

    last_dir = dirs[-1] if dirs else None
    if last_dir is not None:
        last_path = osp.join(last_dir, "last.ckpt")
        if osp.isfile(last_path):
            max_step = ckpts[-1][0] if ckpts else 0
            ckpts.append((max_step + 1, last_path))

    return ckpts


def find_languages(
    languages: list[str], data_dir: str,
) -> list[tuple[str, str]]:
    flores_dir = osp.join(data_dir, "flores")
    found = []
    for lang in languages:
        lang_dir = osp.join(flores_dir, lang)
        if osp.isdir(lang_dir):
            found.append((lang, lang_dir))
        else:
            print(f"  warning: no data for {lang}, skipping", file=sys.stderr)
    return found


def recall_at_k(sim: np.ndarray, k: int) -> float:
    topk = np.argsort(-sim, axis=1)[:, :k]
    correct = np.arange(sim.shape[0])[:, None]
    return float((topk == correct).any(axis=1).mean())


def embed_sentences(
    sentences: list[str],
    encoder,
    renderer,
    transforms,
    patch_size: int,
    max_num_patches: int,
    font_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    layer_buffers: Dict[int, list] = {}
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        processed = [
            preprocess_text_for_classification(
                t, renderer, transforms, patch_size, max_num_patches, font_size,
            )
            for t in batch
        ]
        pv = torch.stack([p["pixel_values"] for p in processed]).to(device)
        am = torch.stack([p["attention_mask"] for p in processed]).to(device)
        batch_embs = embed_batch(encoder, pv, am)
        for k, v in batch_embs.items():
            layer_buffers.setdefault(k, []).append(v)
    return {k: np.vstack(v) for k, v in layer_buffers.items()}


def best_layer(
    metrics: Dict[int, Dict[str, float]], metric_key: str,
) -> tuple[int, Dict[str, float]]:
    idx = max(metrics, key=lambda i: metrics[i][metric_key])
    return idx, metrics[idx]


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_key = f"recall@{args.metric_k}"

    ckpts = collect_checkpoints(args.checkpoint, args.checkpoint_period)
    if not ckpts:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(ckpts)} checkpoint(s)", file=sys.stderr)

    lang_entries = find_languages(args.languages, args.data_dir)
    if len(lang_entries) < 2:
        print("Need at least 2 languages in data directory.", file=sys.stderr)
        sys.exit(1)
    lang_codes = [l for l, _ in lang_entries]
    print(f"{len(lang_entries)} language(s): {lang_codes}", file=sys.stderr)

    pairs = list(itertools.combinations(lang_codes, 2))
    print(f"{len(pairs)} pair(s)", file=sys.stderr)

    font_config = FontConfig(sources=FONTS_NOTO_SANS)
    renderer = _make_renderer(args.renderer, font=font_config)
    transforms = get_transforms(do_resize=False)

    # Load sentences once.
    lang_sentences: Dict[str, list[str]] = {}
    for lang, lang_dir in lang_entries:
        ds = load_from_disk(lang_dir)
        lang_sentences[lang] = [x["sentence"] for x in ds]
    n_per_lang = len(lang_sentences[lang_codes[0]])
    n_langs = len(lang_codes)
    print(f"{n_per_lang} sentences per language", file=sys.stderr)

    all_results = {"args": vars(args), "checkpoints": {}}

    for step, ckpt_path in ckpts:
        ckpt_name = osp.basename(ckpt_path)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"step={step} ({ckpt_name})", file=sys.stderr)

        model, encoder, hparams = load_model_from_checkpoint(
            ckpt_path,
            attn_implementation=args.attn_implementation,
            inference_dtype=TORCH_DTYPES[args.dtype],
        )
        model.to(device).eval()

        patch_size = args.patch_size or hparams.get("patch_size") or 16
        max_num_patches = args.max_num_patches or hparams.get("max_num_patches") or 529

        # Embed each language, then stack into one matrix per layer.
        per_lang_embs: Dict[str, Dict[int, np.ndarray]] = {}
        for lang in lang_codes:
            print(f"  embedding {lang}...", file=sys.stderr)
            per_lang_embs[lang] = embed_sentences(
                lang_sentences[lang], encoder, renderer, transforms,
                patch_size, max_num_patches, args.font_size,
                args.batch_size, device,
            )

        num_layers = len(per_lang_embs[lang_codes[0]])

        # Center globally, normalize, then score pairs (following run_bitext_mining.py).
        ckpt_results = {"step": step, "path": ckpt_path, "pairs": {}}
        pair_metrics_list = []

        # Build per-layer full matrix, center, normalize once.
        centered: Dict[str, Dict[int, np.ndarray]] = {lang: {} for lang in lang_codes}
        for layer_idx in range(num_layers):
            X = np.vstack([per_lang_embs[lang][layer_idx] for lang in lang_codes])

            if args.language_mean_center:
                for i, lang in enumerate(lang_codes):
                    start = i * n_per_lang
                    end = (i + 1) * n_per_lang
                    X[start:end] -= X[start:end].mean(axis=0, keepdims=True)
            else:
                X -= X.mean(axis=0, keepdims=True)

            X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)

            for i, lang in enumerate(lang_codes):
                centered[lang][layer_idx] = X[i * n_per_lang:(i + 1) * n_per_lang]

        # Compute pairwise recall from pre-processed embeddings.
        for lang1, lang2 in pairs:
            pair_key = f"{lang1}-{lang2}"
            pair_results = {}

            for layer_idx in range(num_layers):
                S = centered[lang1][layer_idx] @ centered[lang2][layer_idx].T
                pair_results[layer_idx] = {
                    f"recall@{k}": (recall_at_k(S, k) + recall_at_k(S.T, k)) / 2
                    for k in RECALL_KS
                }

            ckpt_results["pairs"][pair_key] = {
                str(k): v for k, v in pair_results.items()
            }
            pair_metrics_list.append(pair_results)

            bl_idx, bl = best_layer(pair_results, metric_key)
            print(
                f"  {pair_key}: {metric_key}={bl[metric_key]:.3f} (layer {bl_idx})",
                file=sys.stderr,
            )

        if pair_metrics_list:
            avg_metrics = {}
            for layer_idx in range(num_layers):
                avg_metrics[layer_idx] = {
                    mk: float(np.mean([pm[layer_idx][mk] for pm in pair_metrics_list]))
                    for mk in pair_metrics_list[0][layer_idx]
                }
            ckpt_results["average"] = {
                str(k): v for k, v in avg_metrics.items()
            }
            bl_idx, bl = best_layer(avg_metrics, metric_key)
            print(
                f"  avg: {metric_key}={bl[metric_key]:.3f} (layer {bl_idx})",
                file=sys.stderr,
            )

        all_results["checkpoints"][step] = ckpt_results

        del model, encoder, per_lang_embs, centered
        torch.cuda.empty_cache()

    if args.output_path:
        output_path = osp.abspath(osp.expanduser(args.output_path))
        output_dir = osp.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {output_path}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bitext retrieval evaluation on FLORES (offline data).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default=None,
                   help="Model name saved in the output JSON (for plotting / bookkeeping)")
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="Checkpoint .ckpt file(s) or directory(ies) containing them")
    p.add_argument("--checkpoint-period", type=int, default=None,
                   help="Only evaluate checkpoints whose step is a multiple of this value")
    p.add_argument("--languages", nargs="+", required=True,
                   help="FLORES language codes (all unique pairs are evaluated)")
    p.add_argument("--data-dir", default=get_data_dir(),
                   help="Root data directory (expects flores/ subdirectory)")
    p.add_argument("--renderer", required=True,
                   choices=["continuous", "bigrams", "adaptive", "words"])
    p.add_argument("--output-path", default=None,
                   help="Save full JSON results to this path")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--max-num-patches", type=int, default=None)
    p.add_argument("--font-size", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["eager", "sdpa"])
    p.add_argument("--dtype", default="float32", choices=list(TORCH_DTYPES))
    p.add_argument("--language-mean-center", action="store_true",
                   help="Subtract per-language means instead of global mean")
    p.add_argument("--metric-k", type=int, default=5,
                   help="K for recall@K used to select best layer on screen")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
