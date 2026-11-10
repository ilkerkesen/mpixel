"""Convert Universal NER (IOB2) treebanks to HuggingFace datasets saved on disk.

Mirrors the on-disk format of the Naamapadam/KLUE download scripts: each treebank
becomes a DatasetDict (train / validation / test) with `tokens` and a `ner_tags`
ClassLabel, plus a metadata.json. The label set is the fixed canonical UNER scheme
so the id<->label mapping is identical across every treebank.
"""
import argparse
import json
import os
import os.path as osp
import sys
from glob import glob

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from tqdm import tqdm


LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
# Universal NER file names use the UD split names; map "dev" to "validation" to
# match the Naamapadam/KLUE splits.
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
FEATURES = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=LABELS)),
})


def read_iob2(path: str):
    """Read a CoNLL-U-Plus .iob2 file into a list of (tokens, tags) sentences.

    Token rows are tab-separated as `index \t token \t ner_tag \t ... `; `#`
    comment lines are ignored and blank lines separate sentences.
    """
    examples = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.rstrip("\r\n")
            if not line.strip():
                if tokens:
                    examples.append((tokens, tags))
                    tokens, tags = [], []
                continue
            cols = line.split("\t")
            tokens.append(cols[1])
            tags.append(cols[2])
        if tokens:
            examples.append((tokens, tags))
    return examples


def to_dataset(examples) -> Dataset:
    return Dataset.from_dict(
        {
            "tokens": [toks for toks, _ in examples],
            "ner_tags": [[LABEL2ID[t] for t in tags] for _, tags in examples],
        },
        features=FEATURES,
    )


def write_metadata(dataset, dataset_dir: str, treebank: str, udcode: str) -> None:
    metadata = {
        "dataset_name": "universal-ner",
        "treebank": treebank,
        "language": udcode.split("_")[0],
        "labels": LABELS,
        "num_labels": len(LABELS),
        "splits": list(dataset.keys()),
        "num_examples": {split: len(dataset[split]) for split in dataset},
    }
    with open(osp.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def convert_treebank(treebank_dir: str, output_dir: str) -> None:
    treebank = osp.basename(treebank_dir).removeprefix("UNER_").lower().replace("_", "-")
    dataset_dir = osp.join(output_dir, f"universal-ner-{treebank}")
    if osp.isdir(dataset_dir):
        print(f"  {treebank}: already exists, skipping", file=sys.stderr)
        return

    splits, udcode = {}, None
    for src_split, dst_split in SPLITS.items():
        matches = glob(osp.join(treebank_dir, f"*-ud-{src_split}.iob2"))
        if not matches:
            continue
        udcode = osp.basename(matches[0]).split("-ud-")[0]
        splits[dst_split] = to_dataset(read_iob2(matches[0]))

    if not splits:
        return

    dataset = DatasetDict(splits)
    dataset.save_to_disk(dataset_dir)
    write_metadata(dataset, dataset_dir, treebank, udcode)
    print(f"  {treebank}: saved to {dataset_dir}", file=sys.stderr)


def preprocess_universal_ner(input_dir: str, output_dir: str) -> None:
    input_dir = osp.abspath(osp.expanduser(input_dir))
    output_dir = osp.abspath(osp.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    treebank_dirs = sorted(d for d in glob(osp.join(input_dir, "*")) if osp.isdir(d))
    for treebank_dir in tqdm(treebank_dirs, desc="Converting Universal NER"):
        convert_treebank(treebank_dir, output_dir)

    print(f"\nAll Universal NER treebanks saved under {output_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert Universal NER (IOB2) treebanks to HuggingFace datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Root dir containing the UNER_* treebank subdirs",
    )
    p.add_argument("--output-dir", default="./data/ner")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    preprocess_universal_ner(args.input_dir, args.output_dir)
