#!/usr/bin/env python3
"""Promote a YOLOv8 ``.pt`` checkpoint into the model registry."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def promote_yolo(
    *,
    checkpoint: Path,
    version: str,
    dataset_version: str,
    dataset_hash: str,
    map50: float | None,
    registry_root: Path,
) -> Path:
    """Copy YOLO weights into registry and write metadata."""
    out_dir = registry_root / "yolo" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolov8_tumor.pt"
    shutil.copy2(checkpoint, out_path)
    metadata = {
        "model_type": "yolov8",
        "version": version,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "metrics": {"mAP@0.5": map50},
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "weights": out_path.name,
        "source_checkpoint": str(checkpoint),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0.0")
    parser.add_argument("--dataset-version", type=str, default="v1")
    parser.add_argument("--dataset-hash", type=str, default="")
    parser.add_argument("--map50", type=float, default=None)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=_PROJECT_ROOT / "registry",
    )
    args = parser.parse_args(argv)
    out = promote_yolo(
        checkpoint=args.checkpoint,
        version=args.version,
        dataset_version=args.dataset_version,
        dataset_hash=args.dataset_hash,
        map50=args.map50,
        registry_root=args.registry_root,
    )
    print(f"Promoted YOLO to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
