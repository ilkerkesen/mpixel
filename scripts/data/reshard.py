"""Split large JSONL(.gz) files into smaller shards, streaming line by line."""

import argparse
import gzip
import json
import os
import re
import sys
import glob


def _estimate_total(filepaths):
    first = filepaths[0]
    opener = gzip.open if first.endswith(".gz") else open
    count = 0
    with opener(first, "rt") as f:
        for line in f:
            if line.strip():
                count += 1
    return count * len(filepaths)


def reshard(input_dir, output_dir, num_shards, trim_partial_words=False):
    jsonl_files = sorted(
        glob.glob(os.path.join(input_dir, "*.jsonl.gz"))
        + glob.glob(os.path.join(input_dir, "*.jsonl"))
    )

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)

    metadata_files = {
        os.path.basename(mf): mf
        for mf in glob.glob(os.path.join(input_dir, "*-metadata.json"))
    }

    groups = {}
    for filepath in jsonl_files:
        basename = os.path.basename(filepath)
        m = re.match(r"^(.*?)-(\d+)(\.jsonl(?:\.gz)?)$", basename)
        if m:
            prefix, _, ext = m.groups()
        else:
            prefix = os.path.splitext(os.path.splitext(basename)[0])[0]
            ext = ".jsonl.gz" if basename.endswith(".gz") else ".jsonl"
        groups.setdefault((prefix, ext), []).append(filepath)

    for (prefix, ext), filepaths in sorted(groups.items()):
        filepaths = sorted(filepaths)
        out_prefix = prefix

        src_meta = f"{prefix}-metadata.json"
        meta = None
        if src_meta in metadata_files:
            with open(metadata_files[src_meta]) as f:
                meta = json.load(f)

        estimated_total = _estimate_total(filepaths)
        examples_per_shard = max(1, (estimated_total + num_shards - 1) // num_shards)
        print(f"{out_prefix}: ~{estimated_total} examples (estimated) -> ~{num_shards} shards ({examples_per_shard} per shard)", file=sys.stderr)

        out_idx = 0
        out_count = 0
        out_f = None
        total = 0

        for filepath in filepaths:
            opener = gzip.open if filepath.endswith(".gz") else open
            with opener(filepath, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if trim_partial_words:
                        record = json.loads(line)
                        if record.get("chunk-id", 0) > 0:
                            text = record["text"]
                            idx = text.find(' ')
                            if idx != -1:
                                record["text"] = text[idx + 1:]
                            line = json.dumps(record, ensure_ascii=False)

                    if out_f is None:
                        out_path = os.path.join(output_dir, f"{out_prefix}-{out_idx:08d}{ext}")
                        out_f = gzip.open(out_path, "wt") if ext == ".jsonl.gz" else open(out_path, "w")

                    out_f.write(line + "\n")
                    out_count += 1
                    total += 1

                    if out_count >= examples_per_shard:
                        out_f.close()
                        print(f"  {out_prefix}-{out_idx:08d}{ext}: {out_count} examples", file=sys.stderr)
                        out_idx += 1
                        out_count = 0
                        out_f = None

        if out_f is not None:
            out_f.close()
            print(f"  {out_prefix}-{out_idx:08d}{ext}: {out_count} examples", file=sys.stderr)

        actual_shards = out_idx + (1 if out_count > 0 else 0)
        print(f"{out_prefix}: {total} examples -> {actual_shards} shards", file=sys.stderr)

        if meta is not None:
            meta["num_examples"] = total
            meta["num_shards"] = actual_shards
            meta.pop("created_with_another_split", None)
            dst_meta = os.path.join(output_dir, f"{out_prefix}-metadata.json")
            with open(dst_meta, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  wrote {os.path.basename(dst_meta)}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large JSONL shards into smaller ones.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("-N", "--num-shards", type=int, required=True,
                        help="Number of output shards")
    parser.add_argument("--trim-partial-words", action="store_true",
                        help="Remove leading partial word from continuation chunks (chunk-id > 0)")
    args = parser.parse_args()

    reshard(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        trim_partial_words=args.trim_partial_words,
    )
