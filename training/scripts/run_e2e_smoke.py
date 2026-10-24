#!/usr/bin/env python3
"""Orchestrate local E2E smoke: download -> subsample -> 1-epoch Faster R-CNN + YOLO.

Does not promote to registry/v1.0.0. Outputs land under training/outputs/e2e_smoke/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.scripts.download_dataset import download_and_normalize, is_valid_layout
from training.scripts.subsample_dataset import subsample_dataset
from training.scripts.verify_dataset import write_checksum

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_faster_rcnn_smoke(*, smoke_root: Path, output_dir: Path, epochs: int) -> Path:
    """Train Faster R-CNN for a few epochs on the smoke set."""
    from training.scripts.train_faster_rcnn import main as train_faster_rcnn_main

    code = train_faster_rcnn_main(
        [
            "--images",
            str(smoke_root / "train" / "images"),
            "--labels",
            str(smoke_root / "train" / "labels"),
            "--val-images",
            str(smoke_root / "valid" / "images"),
            "--val-labels",
            str(smoke_root / "valid" / "labels"),
            "--epochs",
            str(epochs),
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    if code != 0:
        raise RuntimeError(f"Faster R-CNN train failed with code {code}")
    return output_dir / "best_model.pth"


def _run_yolo_smoke(*, smoke_root: Path, output_dir: Path, epochs: int) -> None:
    """Train YOLOv8 for a few epochs on the smoke set."""
    from training.scripts.train_yolo import main as train_yolo_main

    code = train_yolo_main(
        [
            "--data",
            str(smoke_root / "brain-tumor.yaml"),
            "--epochs",
            str(epochs),
            "--batch",
            "4",
            "--project",
            str(output_dir),
            "--name",
            "tumor",
        ]
    )
    if code != 0:
        raise RuntimeError(f"YOLO train failed with code {code}")


def _run_faster_rcnn_eval(*, checkpoint: Path, smoke_root: Path) -> None:
    """Best-effort Faster R-CNN evaluate on smoke val."""
    from training.scripts.evaluate import main as evaluate_main

    code = evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--images",
            str(smoke_root / "valid" / "images"),
            "--labels",
            str(smoke_root / "valid" / "labels"),
            "--batch-size",
            "2",
        ]
    )
    if code != 0:
        print(f"Faster R-CNN evaluate returned {code} (continuing)")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1",
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1-smoke",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "training" / "outputs" / "e2e_smoke",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require existing data/raw/v1 instead of downloading",
    )
    parser.add_argument("--faster-rcnn-only", action="store_true")
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    if args.faster_rcnn_only and args.yolo_only:
        print("Choose at most one of --faster-rcnn-only / --yolo-only", file=sys.stderr)
        return 2

    # 1) Download / normalize
    if args.skip_download:
        if not is_valid_layout(args.dataset_dir):
            print(
                f"Missing/invalid dataset at {args.dataset_dir}. "
                "Run without --skip-download or run download_dataset.py first.",
                file=sys.stderr,
            )
            return 1
        print(f"Using existing dataset {args.dataset_dir}")
    else:
        download_and_normalize(dest=args.dataset_dir)
        checksum_path, _ = write_checksum(args.dataset_dir)
        print(f"Wrote {checksum_path}")

    # 2) Subsample
    subsample_dataset(
        source=args.dataset_dir,
        dest=args.smoke_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )

    if not args.yolo_only:
        print("=== Faster R-CNN smoke train ===")
        best = _run_faster_rcnn_smoke(
            smoke_root=args.smoke_dir,
            output_dir=args.output_dir / "faster_rcnn",
            epochs=args.epochs,
        )
        if not args.skip_eval and best.exists():
            print("=== Faster R-CNN smoke eval ===")
            _run_faster_rcnn_eval(checkpoint=best, smoke_root=args.smoke_dir)

    if not args.faster_rcnn_only:
        print("=== YOLO smoke train ===")
        _run_yolo_smoke(
            smoke_root=args.smoke_dir,
            output_dir=args.output_dir / "yolo",
            epochs=args.epochs,
        )

    print(
        """
E2E smoke complete.

Next steps (full train / promote - not run automatically):
  python -m training.scripts.train_faster_rcnn --epochs 10
  python -m training.scripts.train_yolo --epochs 50
  python -m training.scripts.promote_checkpoint --model faster_rcnn \\
      --checkpoint training/outputs/faster_rcnn/best_model.pth \\
      --version v1.1.0
  # or reuse HF weights for serve:
  python -m training.scripts.promote_from_hf --revision <commit-sha>
""".strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
