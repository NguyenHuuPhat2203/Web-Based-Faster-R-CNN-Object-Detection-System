#!/usr/bin/env python3
"""Create a deterministic smoke subsample from data/raw/v1 into data/raw/v1-smoke."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from training.data.layout import list_images, write_dataset_yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _copy_pairs(*, src_split: Path, dest_split: Path, images: list[Path]) -> None:
    """Copy image+label pairs; a missing label becomes an empty (background) one."""
    dest_images = dest_split / "images"
    dest_labels = dest_split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, dest_images / image_path.name)
        label_src = src_split / "labels" / f"{image_path.stem}.txt"
        label_dst = dest_labels / label_src.name
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.write_text("", encoding="utf-8")


def subsample_dataset(
    *,
    source: Path,
    dest: Path,
    train_count: int = 16,
    val_count: int = 8,
    seed: int = 42,
) -> Path:
    """Create a smoke subsample dataset directory.

    Train and valid are sampled from their own source splits, so the subsample
    keeps the source split boundary (no train/valid leakage is introduced).

    Raises:
        FileNotFoundError: If the source is missing images in either split.
        ValueError: If more images are requested than exist, or ``dest`` would
            overwrite ``source``.
    """
    if dest.resolve() == source.resolve():
        raise ValueError("--dest must differ from --source (it is deleted first)")
    train_images = list_images(source / "train" / "images")
    val_images = list_images(source / "valid" / "images")
    if not train_images or not val_images:
        raise FileNotFoundError(
            f"Source dataset incomplete: {source}. Run download_dataset.py first."
        )
    if train_count > len(train_images) or val_count > len(val_images):
        raise ValueError(
            f"Requested train={train_count}/val={val_count} but source has "
            f"train={len(train_images)}/val={len(val_images)}"
        )

    rng = random.Random(seed)
    train_pick = sorted(rng.sample(train_images, train_count))
    val_pick = sorted(rng.sample(val_images, val_count))

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    _copy_pairs(
        src_split=source / "train", dest_split=dest / "train", images=train_pick
    )
    _copy_pairs(src_split=source / "valid", dest_split=dest / "valid", images=val_pick)
    write_dataset_yaml(dest / "brain-tumor.yaml")
    print(
        f"Smoke dataset -> {dest} "
        f"(train={len(train_pick)}, valid={len(val_pick)}, seed={seed})"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1-smoke",
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        subsample_dataset(
            source=args.source,
            dest=args.dest,
            train_count=args.train_count,
            val_count=args.val_count,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
