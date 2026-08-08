#!/usr/bin/env python3
"""Train Faster R-CNN from a YOLO-format dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ml.config import DataConfig, ModelConfig, TrainConfig
from training.data.dataset import YoloDetectionDataset, detection_collate
from training.trainer.trainer import train_faster_rcnn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1" / "train" / "images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1" / "train" / "labels",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
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
        data=DataConfig(batch_size=args.batch_size, dataset_version="v1"),
        model=ModelConfig(),
    )
    dataset = YoloDetectionDataset(args.images, args.labels)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
    )
    best = train_faster_rcnn(
        loader,
        config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
