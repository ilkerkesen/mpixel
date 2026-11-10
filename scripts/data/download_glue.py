"""Download GLUE benchmark datasets and save to disk for offline HPC use."""
import argparse
import os
import os.path as osp
import sys

from datasets import load_dataset
from tqdm import tqdm

from mpixel.utils import get_data_dir


GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


def download_glue(tasks: list[str], output_dir: str) -> None:
    dataset_dir = osp.join(osp.abspath(osp.expanduser(output_dir)), "glue")
    os.makedirs(dataset_dir, exist_ok=True)

    for task in tqdm(tasks, desc="Downloading GLUE"):
        task_dir = osp.join(dataset_dir, task)
        if osp.isdir(task_dir):
            print(f"  {task}: already exists, skipping", file=sys.stderr)
            continue

        print(f"  Downloading {task}...", file=sys.stderr)
        dataset = load_dataset("glue", task)
        dataset.save_to_disk(task_dir)
        print(f"  {task}: saved to {task_dir}", file=sys.stderr)

    print(f"\nAll GLUE datasets saved to {dataset_dir}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download GLUE benchmark datasets for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tasks", nargs="+", default=GLUE_TASKS,
        help="GLUE task names to download",
    )
    p.add_argument("--output-dir", default=get_data_dir())
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    download_glue(args.tasks, args.output_dir)
