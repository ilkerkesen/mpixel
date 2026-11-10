import os
import os.path as osp
import sys
import gc
import argparse
from pathlib import Path

from datasets import load_dataset

from mpixel.data.fineweb import (
    CompressionType,
    create_measure_unit_patches_fn,
    split_text_by_patches,
    _stream_to_shards,
)


def stream_fineweb2(
    language: str,
    num_train_patches: int = 0,
    split: str = "train",
    text_column: str = "text",
    output_dir: str | Path = ".",
    patches_per_file: int = 10_000,
    compression: CompressionType = "gzip",
    num_patches: int = 1024,
    seed: int = 4477,
    block_size: int = 16,
    font_size: int = 12,
    pad: int | tuple[int, int] = 1,
) -> None:

    if patches_per_file <= 0:
        raise ValueError("patches_per_file must be positive")

    output_dir = Path(osp.abspath(output_dir))

    dataset_name = 'HuggingFaceFW/fineweb-2' if language != 'eng_Latn' else 'HuggingFaceFW/fineweb'
    dataset_config = 'sample-100BT' if language == 'eng_Latn' else language

    measure_fn = create_measure_unit_patches_fn(
        block_size=block_size,
        font_size=font_size,
        pad=pad,
    )

    print(
        f"Streaming dataset '{dataset_name}' (split='{split}')\n"
        f"  language = {language}\n"
        f"  num_train_patches = {num_train_patches} (0=skip)\n"
        f"  patches per file = {patches_per_file}\n"
        f"  compression = {compression}\n"
        f"  output dir = {output_dir}",
        file=sys.stderr,
    )

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=1_000)
    iterator = iter(shuffled_dataset)

    metadata = {
        "format": "welt-preprocessed-v2",
        "source_dataset": dataset_name,
        "source_config": dataset_config,
        "source_split": split,
        "language": language,
        "max_seq_length": num_patches,
        "seed": seed,
        "text_column": text_column,
        "text_template": None,
        "split": "train",
    }

    source_short = 'fineweb2' if language != 'eng_Latn' else 'fineweb'

    try:
        if num_train_patches > 0:
            train_prefix = f"{source_short}-{language}-train"
            _stream_to_shards(
                iterator,
                max_patches=num_train_patches,
                output_dir=output_dir,
                file_prefix=train_prefix,
                split_name='train',
                metadata=metadata,
                language=language,
                patches_per_file=patches_per_file,
                compression=compression,
                measure_fn=measure_fn,
                num_patches=num_patches,
                text_column=text_column,
            )
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.", file=sys.stderr)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stream FineWeb2 by language into sharded JSONL(.gz)."
    )

    parser.add_argument("language")
    parser.add_argument("--num-train-patches", type=int, default=0,
                        help="Max total patches for training (0=skip)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="data/pretraining/v1")
    parser.add_argument("--patches-per-file", type=int, default=10_000)
    parser.add_argument("--compression", choices=["none", "gzip", "zstd"], default="zstd")
    parser.add_argument("--num-patches", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4477)

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    stream_fineweb2(
        language=args.language,
        num_train_patches=args.num_train_patches,
        split=args.split,
        text_column=args.text_column,
        output_dir=args.output_dir,
        patches_per_file=args.patches_per_file,
        compression=args.compression,
        num_patches=args.num_patches,
        seed=args.seed,
    )
    gc.collect()
    os._exit(0)
