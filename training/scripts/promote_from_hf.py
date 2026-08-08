#!/usr/bin/env python3
"""Download Faster R-CNN weights from Hugging Face and promote into the registry.

YOLO is promoted only when ``checkpoints/yolov8_tumor.pt`` already exists locally.
Weights remain gitignored; only ``metadata.json`` should be committed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from training.scripts.export_safetensors import export_and_promote
from training.scripts.promote_yolo import promote_yolo


def _download_frcnn_checkpoint(
    *,
    dest: Path,
    repo_id: str,
    filename: str,
    token: str | None,
) -> Path:
    """Download HF checkpoint into ``dest`` (copied from hub cache)."""
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    )
    shutil.copy2(cached, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="LuxeFats/FasterRCNN-Checkpoint",
        help="Hugging Face repository id",
    )
    parser.add_argument(
        "--filename",
        default="model.pth",
        help="Checkpoint filename inside the HF repo",
    )
    parser.add_argument("--token", default=None, help="Optional HF access token")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=_PROJECT_ROOT / "registry",
    )
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-hash", default="")
    parser.add_argument("--map50", type=float, default=0.69)
    parser.add_argument("--map5095", type=float, default=0.49)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing checkpoints/model.pth without calling Hugging Face",
    )
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint_dir / "model.pth"
    if args.skip_download:
        if not checkpoint_path.exists():
            print(f"Missing checkpoint: {checkpoint_path}", file=sys.stderr)
            return 1
    else:
        print(f"Downloading {args.repo_id}/{args.filename} ...")
        _download_frcnn_checkpoint(
            dest=checkpoint_path,
            repo_id=args.repo_id,
            filename=args.filename,
            token=args.token,
        )
        print(f"Saved {checkpoint_path}")

    out = export_and_promote(
        checkpoint=checkpoint_path,
        version=args.version,
        dataset_version=args.dataset_version,
        dataset_hash=args.dataset_hash,
        map50=args.map50,
        map5095=args.map5095,
        registry_root=args.registry_root,
    )
    print(f"Promoted Faster R-CNN to {out}")

    yolo_src = args.checkpoint_dir / "yolov8_tumor.pt"
    if yolo_src.exists():
        yolo_out = promote_yolo(
            checkpoint=yolo_src,
            version=args.version,
            dataset_version=args.dataset_version,
            dataset_hash=args.dataset_hash,
            map50=None,
            registry_root=args.registry_root,
        )
        print(f"Promoted YOLO to {yolo_out}")
    else:
        print(f"Skipping YOLO promote (not found: {yolo_src})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
