#!/usr/bin/env python3
"""Create a deterministic smoke subsample from data/raw/v1 into data/raw/v1-smoke."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_stems(images_dir: Path) -> list[str]:
    """Return sorted image stems from an images directory."""
    return sorted(
        p.stem
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _copy_pairs(
    *,
    src_split: Path,
    dest_split: Path,
    stems: list[str],
) -> None:
    """Copy image+label pairs for the given stems."""
    src_images = src_split / "images"
    src_labels = src_split / "labels"
    dest_images = dest_split / "images"
    dest_labels = dest_split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        matches = [
            p
            for p in src_images.iterdir()
            if p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not matches:
            continue
        image_path = matches[0]
        shutil.copy2(image_path, dest_images / image_path.name)
        label_src = src_labels / f"{stem}.txt"
        label_dst = dest_labels / f"{stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.write_text("", encoding="utf-8")


def _write_yaml(dest: Path) -> None:
    """Write dataset YAML for the smoke layout."""
    dest.write_text(
        """# Smoke subsample — do not commit images
path: .
train: train/images
val: valid/images

names:
  0: negative
  1: positive
""",
        encoding="utf-8",
    )


def subsample_dataset(
    *,
    source: Path,
    dest: Path,
    train_count: int = 16,
    val_count: int = 8,
    seed: int = 42,
) -> Path:
    """Create a smoke subsample dataset directory."""
    train_stems = _list_stems(source / "train" / "images")
    val_stems = _list_stems(source / "valid" / "images")
    if not train_stems or not val_stems:
        raise FileNotFoundError(
            f"Source dataset incomplete: {source}. Run download_dataset.py first."
        )
    if train_count > len(train_stems) or val_count > len(val_stems):
        raise ValueError(
            f"Requested train={train_count}/val={val_count} but source has "
            f"train={len(train_stems)}/val={len(val_stems)}"
        )

    rng = random.Random(seed)
    train_pick = sorted(rng.sample(train_stems, train_count))
    val_pick = sorted(rng.sample(val_stems, val_count))

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    _copy_pairs(src_split=source / "train", dest_split=dest / "train", stems=train_pick)
    _copy_pairs(src_split=source / "valid", dest_split=dest / "valid", stems=val_pick)
    _write_yaml(dest / "brain-tumor.yaml")
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
