"""Download the KLUE NER dataset (Korean) and save to disk."""
import argparse
import json
import os
import os.path as osp
import sys

from datasets import load_dataset


HF_DATASET_NAME = "klue"
HF_DATASET_CONFIG = "ner"


def write_metadata(dataset, dataset_dir: str) -> None:
    labels = dataset["train"].features["ner_tags"].feature.names
    metadata = {
        "dataset_name": f"{HF_DATASET_NAME}/{HF_DATASET_CONFIG}",
        "language": "ko",
        "labels": labels,
        "num_labels": len(labels),
        "splits": list(dataset.keys()),
        "num_examples": {split: len(dataset[split]) for split in dataset},
    }
    with open(osp.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def download_klue_ner(output_dir: str) -> None:
    base_dir = osp.abspath(osp.expanduser(output_dir))
    dataset_dir = osp.join(base_dir, "klue-ner")
    if osp.isdir(dataset_dir):
        print("klue-ner: already exists, skipping", file=sys.stderr)
        return

    os.makedirs(base_dir, exist_ok=True)
    print(f"Downloading {HF_DATASET_NAME}/{HF_DATASET_CONFIG}...", file=sys.stderr)
    dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG)
    dataset.save_to_disk(dataset_dir)
    write_metadata(dataset, dataset_dir)
    print(f"Saved to {dataset_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download the KLUE NER dataset for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="./data/ner")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    download_klue_ner(args.output_dir)
