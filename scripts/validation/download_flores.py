"""Download FLORES data for bitext retrieval, filtering out SIB-200 examples.

For each language, loads FLORES dev+devtest, removes examples whose index
matches any SIB-200 split (train/val/test), and saves one dataset per language.
Since FLORES is fully aligned, any pair can be constructed at evaluation time
by loading two languages and matching by row index.
"""
import argparse
import json
import os
import os.path as osp
import sys

from datasets import Dataset, load_dataset
from tqdm import tqdm

from mpixel.utils import get_data_dir


DEFAULT_LANGUAGES = [
    "arb_Arab", "ben_Beng", "zho_Hans", "dan_Latn",
    "ell_Grek", "eng_Latn", "pes_Arab", "fin_Latn",
    "heb_Hebr", "hin_Deva", "kor_Hang", "spa_Latn",
    "tha_Thai", "tur_Latn", "ukr_Cyrl", "vie_Latn",
]


def load_flores_sentences(lang: str):
    """Load FLORES dev+devtest for a single language, returning {concat_idx: {id, sentence}}."""
    dev = load_dataset("facebook/flores", lang, split="dev", trust_remote_code=True)
    devtest = load_dataset("facebook/flores", lang, split="devtest", trust_remote_code=True)

    sentences = {}
    for x in dev:
        idx = x["id"] - 1
        sentences[idx] = {"id": f"dev-{x['id']}", "sentence": x["sentence"]}
    for x in devtest:
        idx = len(dev) + x["id"] - 1
        sentences[idx] = {"id": f"devtest-{x['id']}", "sentence": x["sentence"]}

    return sentences


def load_sib200_index_ids(lang: str) -> set[int]:
    """Collect all index_ids from SIB-200 train/val/test for a language."""
    ids = set()
    for split in ["train", "validation", "test"]:
        ds = load_dataset("Davlan/sib200", lang, split=split)
        for x in ds:
            ids.add(x["index_id"])
    return ids


def download_flores(languages: list[str], output_dir: str) -> None:
    dataset_dir = osp.join(osp.abspath(osp.expanduser(output_dir)), "flores")
    os.makedirs(dataset_dir, exist_ok=True)

    print("Collecting SIB-200 index_ids to exclude...", file=sys.stderr)
    sib_ids = load_sib200_index_ids(languages[0])
    print(f"  {len(sib_ids)} SIB-200 examples to filter out", file=sys.stderr)

    print("Loading FLORES and verifying against SIB-200...", file=sys.stderr)
    keep_indices = None

    for lang in tqdm(languages, desc="Downloading"):
        lang_dir = osp.join(dataset_dir, lang)
        if osp.isdir(lang_dir):
            print(f"  {lang}: already exists, skipping", file=sys.stderr)
            continue

        flores = load_flores_sentences(lang)

        sib_ids_lang = load_sib200_index_ids(lang)
        if sib_ids_lang != sib_ids:
            print(
                f"  WARNING: SIB-200 index_ids differ for {lang} "
                f"({len(sib_ids_lang)} vs {len(sib_ids)})",
                file=sys.stderr,
            )

        # Verify text match for SIB-200 examples
        mismatches = 0
        for split in ["train", "validation", "test"]:
            ds = load_dataset("Davlan/sib200", lang, split=split)
            for x in ds:
                idx = x["index_id"]
                if idx in flores and x["text"].strip() != flores[idx]["sentence"].strip():
                    mismatches += 1
        if mismatches > 0:
            print(f"  WARNING: {mismatches} text mismatches for {lang}", file=sys.stderr)

        filtered = {idx: v for idx, v in flores.items() if idx not in sib_ids}

        if keep_indices is None:
            keep_indices = sorted(filtered.keys())
        elif sorted(filtered.keys()) != keep_indices:
            print(f"ERROR: filtered indices differ for {lang}", file=sys.stderr)
            sys.exit(1)

        rows = [
            {"id": filtered[idx]["id"], "sentence": filtered[idx]["sentence"]}
            for idx in keep_indices
        ]

        ds = Dataset.from_list(rows)
        os.makedirs(lang_dir, exist_ok=True)
        ds.save_to_disk(lang_dir)

    print(
        f"\n{len(keep_indices)} examples per language after filtering",
        file=sys.stderr,
    )

    metadata = {
        "languages": languages,
        "num_examples": len(keep_indices),
        "sib200_filtered": len(sib_ids),
    }
    with open(osp.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {dataset_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download FLORES data, filtering out SIB-200 examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGUAGES,
        help="FLORES language codes",
    )
    p.add_argument("--output-dir", default=get_data_dir())
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    download_flores(args.languages, args.output_dir)
