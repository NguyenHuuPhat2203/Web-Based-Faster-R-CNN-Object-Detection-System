#!/usr/bin/env python3
"""Train YOLOv8 via Ultralytics API."""

from __future__ import annotations

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1" / "brain-tumor.yaml",
        help="Ultralytics dataset YAML",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument(
        "--project",
        type=Path,
        default=_PROJECT_ROOT / "training" / "outputs" / "yolo",
    )
    parser.add_argument("--name", type=str, default="tumor")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
