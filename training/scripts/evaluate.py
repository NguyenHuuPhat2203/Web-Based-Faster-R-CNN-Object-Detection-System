#!/usr/bin/env python3
"""Evaluate a Faster R-CNN checkpoint on a YOLO-format validation set (loss only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ml.models.faster_rcnn import create_faster_rcnn, get_device, load_state_dict
from training.data.dataset import YoloDetectionDataset, detection_collate
from training.trainer.trainer import evaluate_loss

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VALID_ROOT = _PROJECT_ROOT / "data" / "raw" / "v1" / "valid"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=_VALID_ROOT / "images")
    parser.add_argument("--labels", type=Path, default=_VALID_ROOT / "labels")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    device = get_device()
    model = create_faster_rcnn(pretrained=False)
    load_state_dict(model, args.checkpoint, device=device)
    model.to(device)

    loader = DataLoader(
        YoloDetectionDataset(args.images, args.labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate,
    )
    val_loss = evaluate_loss(model, loader, device)
    print(f"val_loss={val_loss:.6f} images={len(loader.dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
