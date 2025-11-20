#!/usr/bin/env python3
"""Download TrajCast datasets from HuggingFace Hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from shutil import move

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "ibm-research/trajcast.datasets-arxiv2025"
DEFAULT_REVISION = "main"
DEFAULT_SPLITS = ("train", "val", "test")

SYSTEM_ALIAS = {
    "example": "example",
    "paracetamol": "paracetamol",
    "water": "water",
    "quartz": "quartz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download prepared molecular dynamics datasets used by TrajCast. "
            "This mirrors the workflow from examples/training/training.ipynb."
        )
    )
    parser.add_argument(
        "--dataset",
        default="example",
        choices=sorted(SYSTEM_ALIAS.keys()),
        help="Which dataset subtree to fetch from the HuggingFace repo.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        metavar="SPLIT",
        help="Dataset splits to download (default: train val test).",
    )
    parser.add_argument(
        "--target-dir",
        default="data",
        type=Path,
        help="Directory where the dataset should be stored (default: ./data).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HuggingFace repo id containing the dataset.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset repo revision/tag to download (default: main).",
    )
    parser.add_argument(
        "--extension",
        default="extxyz",
        help="File extension of the remote split files (default: extxyz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser.parse_args()


def download_split(
    *,
    split: str,
    dataset: str,
    repo_id: str,
    revision: str,
    extension: str,
    target_dir: Path,
    overwrite: bool,
) -> Path:
    remote_path = f"{dataset}/{split}.{extension}"
    destination = target_dir / dataset
    destination.mkdir(parents=True, exist_ok=True)
    local_file = destination / f"{split}.{extension}"

    if local_file.exists() and not overwrite:
        print(f"[skip] {local_file} already exists", file=sys.stderr)
        return local_file

    raw_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=remote_path,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
        )
    )

    if raw_path != local_file:
        if local_file.exists():
            local_file.unlink()
        move(str(raw_path), str(local_file))
        nested_parent = raw_path.parent
        try:
            nested_parent.rmdir()
        except OSError:
            pass

    print(f"[done] Downloaded {remote_path} -> {local_file}")
    return local_file


def main() -> None:
    args = parse_args()
    dataset = SYSTEM_ALIAS[args.dataset]
    for split in args.splits:
        download_split(
            split=split,
            dataset=dataset,
            repo_id=args.repo_id,
            revision=args.revision,
            extension=args.extension,
            target_dir=args.target_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
