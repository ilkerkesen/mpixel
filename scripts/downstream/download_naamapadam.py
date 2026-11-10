"""Download the Naamapadam NER dataset (Indic languages) and save to disk."""
import argparse
import json
import os
import os.path as osp
import sys

from datasets import load_dataset
from tqdm import tqdm


HF_DATASET_NAME = "ai4bharat/naamapadam"
LANGUAGES = ["bn", "hi", "ta", "te"]


def write_metadata(dataset, dataset_dir: str, language: str) -> None:
    labels = dataset["train"].features["ner_tags"].feature.names
    metadata = {
        "dataset_name": HF_DATASET_NAME,
        "language": language,
        "labels": labels,
        "num_labels": len(labels),
        "splits": list(dataset.keys()),
        "num_examples": {split: len(dataset[split]) for split in dataset},
    }
    with open(osp.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def download_naamapadam(languages: list[str], output_dir: str) -> None:
    base_dir = osp.abspath(osp.expanduser(output_dir))
    os.makedirs(base_dir, exist_ok=True)

    for language in tqdm(languages, desc="Downloading Naamapadam"):
        dataset_dir = osp.join(base_dir, f"naamapadam-{language}")
        if osp.isdir(dataset_dir):
            print(f"  {language}: already exists, skipping", file=sys.stderr)
            continue

        print(f"  Downloading {HF_DATASET_NAME}/{language}...", file=sys.stderr)
        dataset = load_dataset(HF_DATASET_NAME, language)
        dataset.save_to_disk(dataset_dir)
        write_metadata(dataset, dataset_dir, language)
        print(f"  {language}: saved to {dataset_dir}", file=sys.stderr)

    print(f"\nAll Naamapadam datasets saved under {base_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download the Naamapadam NER dataset for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--languages", nargs="+", default=LANGUAGES,
        help="Naamapadam language codes to download",
    )
    p.add_argument("--output-dir", default="./data/ner")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    download_naamapadam(args.languages, args.output_dir)
