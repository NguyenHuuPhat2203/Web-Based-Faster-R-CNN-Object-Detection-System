#!/usr/bin/env python3
"""Evaluate a Faster R-CNN checkpoint on a YOLO-format validation set (loss only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.backbone.faster_rcnn import create_faster_rcnn, get_device, load_state_dict
from training.data.dataset import YoloDetectionDataset, detection_collate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--images",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1" / "valid" / "images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1" / "valid" / "labels",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    device = get_device()
    model = create_faster_rcnn()
    load_state_dict(model, args.checkpoint, device=device)
    model.to(device)
    # torchvision detection models return losses only in train mode
    model.train()

    dataset = YoloDetectionDataset(args.images, args.labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate,
    )

    total = 0.0
    steps = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = float(sum(loss_dict.values()).item())
        total += loss
        steps += 1

    avg = total / max(steps, 1)
    print(f"val_loss={avg:.6f} steps={steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
