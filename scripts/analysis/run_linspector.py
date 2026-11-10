"""LINSPECTOR word-level (intrinsic) probing analysis.

Extracts mean-pooled hidden states per layer from a pretrained PIXEL model and
trains a logistic-regression probe per layer/task, following the LINSPECTOR
intrinsic probing methodology.

Probing data is expected in the layout cloned from
https://github.com/UKPLab/linspector, i.e.

    <data-dir>/<Task>/<language>/{train,dev,test}.txt

where each line is ``word(s) <tab> label`` (the last whitespace-separated token
is the label). By default ``<data-dir>`` points at
``<repo>/data/linspector/intrinsic/data``.

Example
-------
    for lang in arabic armenian modern-greek macedonian russian; do
        python scripts/analysis/run_linspector.py \
            --checkpoint path/to/model.ckpt --renderer bigrams --lang $lang
    done
    python scripts/analysis/plot_linspector.py \
        --models my-model --results-dir outputs/analysis/linspector
"""
import argparse
import json
import os
import os.path as osp

import torch

from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS

from mpixel.models import auto_register  # noqa: F401
from mpixel.analysis.probe import Probe
from mpixel.datamodules.sib200 import (
    _make_renderer,
    preprocess_text_for_classification,
)
from mpixel.utils import get_data_dir
from mpixel.utils.checkpoint import load_model_from_checkpoint
from mpixel.utils.embedding import embed_batch
from mpixel.utils.preprocessing import get_transforms


TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def make_batcher(encoder, renderer, transforms, patch_size, max_num_patches, font_size, device):
    """Build a ``batcher(list[str]) -> {layer: (B, D) ndarray}`` callable.

    Renders each input string, pads/builds the attention mask, and returns the
    per-layer mean-pooled embeddings via the shared ``embed_batch`` helper.
    """

    def batcher(batch_texts):
        processed = [
            preprocess_text_for_classification(
                t, renderer, transforms, patch_size, max_num_patches, font_size,
            )
            for t in batch_texts
        ]
        pixel_values = torch.stack([p["pixel_values"] for p in processed]).to(device)
        attention_mask = torch.stack([p["attention_mask"] for p in processed]).to(device)
        return embed_batch(encoder, pixel_values, attention_mask)

    return batcher


def make_text_batcher(model, tokenizer, max_length, device):
    """Build a ``batcher(list[str]) -> {layer: (B, D) ndarray}`` for HF text models.

    Tokenizes each input string and returns the per-layer mean-pooled (over
    non-pad tokens) hidden states, mirroring the pixel ``embed_batch`` pooling.
    """

    @torch.no_grad()
    def batcher(batch_texts):
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        attention_mask = enc["attention_mask"]
        embeddings = {}
        for layer_idx, h in enumerate(outputs.hidden_states):
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
            embeddings[layer_idx] = pooled.float().cpu().numpy()
        return embeddings

    return batcher


def discover_tasks(data_dir, lang):
    """Return probing tasks that have data for ``lang`` (skipping Character* tasks)."""
    tasks = []
    for task in sorted(os.listdir(data_dir)):
        full_path = osp.join(data_dir, task, lang)
        if osp.isdir(full_path) and not task.startswith("Character"):
            tasks.append(task)
    return tasks


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if osp.isdir(args.checkpoint):
        # HuggingFace text model: tokenize inputs instead of rendering them.
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModel.from_pretrained(args.checkpoint)
        model.to(device=device, dtype=TORCH_DTYPES[args.dtype]).eval()

        model_variant = model.config.model_type
        print(f"variant={model_variant} (huggingface text) max_length={args.max_length}")

        batcher = make_text_batcher(model, tokenizer, args.max_length, device)
        run_name = args.run_name or osp.basename(osp.abspath(args.checkpoint.rstrip("/")))
        args_record = {
            "checkpoint": args.checkpoint,
            "lang": args.lang,
            "input_type": "text",
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "layer": args.layer,
            "seed": args.seed,
            "model_variant": model_variant,
        }
    else:
        model, encoder, hparams = load_model_from_checkpoint(
            args.checkpoint,
            attn_implementation=args.attn_implementation,
            inference_dtype=TORCH_DTYPES[args.dtype],
        )
        model.to(device).eval()

        patch_size = args.patch_size or hparams.get("patch_size") or 16
        max_num_patches = args.max_num_patches or hparams.get("max_num_patches") or 529

        print(f"variant={hparams.get('model_variant')} patch_size={patch_size} "
              f"max_num_patches={max_num_patches} renderer={args.renderer} font_size={args.font_size}")

        renderer = _make_renderer(args.renderer, font=FontConfig(sources=FONTS_NOTO_SANS))
        transforms = get_transforms(do_resize=False)
        batcher = make_batcher(
            encoder, renderer, transforms, patch_size, max_num_patches, args.font_size, device,
        )
        run_name = args.run_name or osp.basename(osp.dirname(osp.dirname(osp.abspath(args.checkpoint))))
        args_record = {
            "checkpoint": args.checkpoint,
            "lang": args.lang,
            "input_type": "pixel",
            "renderer": args.renderer,
            "patch_size": patch_size,
            "max_num_patches": max_num_patches,
            "font_size": args.font_size,
            "batch_size": args.batch_size,
            "layer": args.layer,
            "seed": args.seed,
            "model_variant": hparams.get("model_variant"),
        }

    layers = None if args.layer == "all" else [int(args.layer)]

    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    tasks = args.tasks or discover_tasks(data_dir, args.lang)
    print(f"lang={args.lang} tasks = {', '.join(tasks)}")

    all_results = {
        "args": args_record,
        "lang": args.lang,
        "run_name": run_name,
        "results": {},
    }

    for task in tasks:
        probe = Probe(task=task, lang=args.lang, data_dir=data_dir, seed=args.seed)
        task_results = probe.run(batcher, batch_size=args.batch_size, layers=layers)

        all_results["results"][task] = {
            str(layer): {"dev_acc": float(devacc), "test_acc": float(testacc)}
            for layer, (devacc, testacc, _predictions) in task_results.items()
        }

    output_dir = osp.abspath(osp.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f"{run_name}-{args.lang}-linspector.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a PyTorch Lightning .ckpt file, or a HuggingFace model "
                        "directory (loaded as a text model with tokenized input)")
    p.add_argument("--renderer", default="adaptive",
                   choices=["continuous", "bigrams", "adaptive", "words"],
                   help="Renderer used during pretraining (pixel checkpoints only)")
    p.add_argument("--lang", required=True, help="Probing language (subdirectory name)")
    p.add_argument("--data-dir", default=osp.join(get_data_dir(), "linspector", "intrinsic", "data"),
                   help="Directory containing <Task>/<lang>/{train,dev,test}.txt")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Probing tasks to run (default: all available, excluding Character*)")
    p.add_argument("--layer", default="all", help="Layer index to probe, or 'all'")
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--max-num-patches", type=int, default=None)
    p.add_argument("--font-size", type=int, default=12)
    p.add_argument("--max-length", type=int, default=128,
                   help="Max tokens per input for HuggingFace text models")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa"])
    p.add_argument("--dtype", default="float32", choices=list(TORCH_DTYPES))
    p.add_argument("--run-name", default=None,
                   help="Name used in output filenames (default: inferred from checkpoint path)")
    p.add_argument("--output-dir", default="outputs/analysis/linspector")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
