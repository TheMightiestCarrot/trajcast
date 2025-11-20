#!/usr/bin/env python3
"""
Subsample ASE-extxyz trajectories by uniform stride to create a smaller dataset.

Usage (examples):
  # keep ~20% into a sibling folder paracetamol_reduced with same filenames+_sub
  python scripts/subsample_extxyz.py --ratio 0.2 --out-dir data/paracetamol_reduced \
      data/paracetamol/train.extxyz data/paracetamol/val.extxyz

  python scripts/subsample_extxyz.py --stride 10 --suffix _small \
      data/paracetamol/train.extxyz data/paracetamol/val.extxyz data/paracetamol/test.extxyz

Notes:
- Targets (displacements, update_velocities) are stored per frame, so striding
  keeps them consistent; the physical horizon remains the original timestep.
- Output files are written next to the inputs with a configurable suffix
  (default: "_sub"). Existing outputs are overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase.io


def subsample_one(src: Path, out: Path, stride: int) -> tuple[int, int]:
    """Write every `stride`-th frame from `src` to `out` (inclusive of frame 0)."""
    if stride < 1:
        raise ValueError("stride must be >= 1")

    if out.exists():
        out.unlink()

    kept = 0
    total = 0
    for total, atoms in enumerate(ase.io.iread(src, index=":"), start=1):
        if (total - 1) % stride == 0:
            ase.io.write(out, atoms, format="extxyz", append=out.exists())
            kept += 1

    return total, kept


def main():
    parser = argparse.ArgumentParser(description="Subsample extxyz trajectories by stride.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ratio", type=float, help="Fraction of frames to keep (e.g., 0.2).")
    group.add_argument("--stride", type=int, help="Keep every Nth frame (overrides ratio).")
    parser.add_argument(
        "--suffix",
        default="_sub",
        help="Suffix added before the file extension for output files (default: _sub).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to place outputs. If unset, files are written next to inputs.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Input extxyz files to subsample.")

    args = parser.parse_args()

    stride = args.stride
    if stride is None:
        if not (0 < args.ratio <= 1):
            raise ValueError("ratio must be in (0, 1].")
        stride = max(1, round(1 / args.ratio))

    out_dir = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.files:
        if not src.exists():
            raise FileNotFoundError(src)
        if out_dir:
            out = out_dir / (src.stem + args.suffix + src.suffix)
        else:
            out = src.with_name(src.stem + args.suffix + src.suffix)
        total, kept = subsample_one(src, out, stride)
        print(f"{src}: kept {kept}/{total} frames (stride {stride}) -> {out}")


if __name__ == "__main__":
    main()
