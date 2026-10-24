#!/usr/bin/env python3
"""Train Faster R-CNN from a YOLO-format dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ml.config import DataConfig, TrainConfig
from training.data.dataset import YoloDetectionDataset, detection_collate
from training.trainer.trainer import train_faster_rcnn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET_ROOT = _PROJECT_ROOT / "data" / "raw" / "v1"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=_DATASET_ROOT / "train" / "images"
    )
    parser.add_argument(
        "--labels", type=Path, default=_DATASET_ROOT / "train" / "labels"
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=_DATASET_ROOT / "valid" / "images",
        help="Validation images; validation loss selects the best checkpoint",
    )
    parser.add_argument(
        "--val-labels", type=Path, default=_DATASET_ROOT / "valid" / "labels"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "training" / "outputs" / "faster_rcnn",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    config = TrainConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        data=DataConfig(batch_size=args.batch_size, num_workers=args.num_workers),
    )
    train_loader = DataLoader(
        YoloDetectionDataset(args.images, args.labels, config.model.num_classes),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        YoloDetectionDataset(
            args.val_images, args.val_labels, config.model.num_classes
        ),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
    )
    best = train_faster_rcnn(
        train_loader,
        val_loader,
        config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
