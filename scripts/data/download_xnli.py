"""Download XNLI datasets (all 15 languages) and save to disk for offline HPC use."""
import argparse
import os
import os.path as osp
import sys

from datasets import load_dataset
from tqdm import tqdm

from mpixel.utils import get_data_dir


XNLI_LANGUAGES = ["en", "fr", "es", "el", "de", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]


def download_xnli(languages: list[str], output_dir: str) -> None:
    dataset_dir = osp.join(osp.abspath(osp.expanduser(output_dir)), "xnli")
    os.makedirs(dataset_dir, exist_ok=True)

    for language in tqdm(languages, desc="Downloading XNLI"):
        lang_dir = osp.join(dataset_dir, language)
        if osp.isdir(lang_dir):
            print(f"  {language}: already exists, skipping", file=sys.stderr)
            continue

        print(f"  Downloading {language}...", file=sys.stderr)
        dataset = load_dataset("xnli", language)
        dataset.save_to_disk(lang_dir)
        print(f"  {language}: saved to {lang_dir}", file=sys.stderr)

    print(f"\nAll XNLI datasets saved to {dataset_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download XNLI datasets for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--languages", nargs="+", default=XNLI_LANGUAGES,
        help="XNLI language configs to download",
    )
    p.add_argument("--output-dir", default=get_data_dir())
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    download_xnli(args.languages, args.output_dir)
