"""Dataset layout shared by download, subsample, and training code."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

# No ``path:`` key: Ultralytics then resolves splits relative to this YAML file.
# A ``path: .`` would resolve against the Ultralytics datasets dir instead.
_DATASET_YAML = """# Brain-tumor dataset (normalized for this repo)
# Source: https://docs.ultralytics.com/datasets/detect/brain-tumor/
train: train/images
val: valid/images

names:
  0: negative
  1: positive
"""


def list_images(images_dir: Path) -> list[Path]:
    """Return image files in ``images_dir``, sorted by name."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_dataset_yaml(dest: Path) -> None:
    """Write the Ultralytics dataset YAML for the normalized layout."""
    dest.write_text(_DATASET_YAML, encoding="utf-8")
