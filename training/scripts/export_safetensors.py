#!/usr/bin/env python3
"""Export a ``.pth`` checkpoint to ``.safetensors`` and promote to the registry."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_file

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


def export_and_promote(
    *,
    checkpoint: Path,
    version: str,
    dataset_version: str,
    dataset_hash: str,
    map50: float | None,
    map5095: float | None,
    registry_root: Path,
) -> Path:
    """Convert checkpoint to safetensors and write registry metadata.

    Returns:
        Path to the promoted ``model.safetensors``.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    out_dir = registry_root / "faster_rcnn" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.safetensors"
    # safetensors requires contiguous CPU tensors
    clean = {k: v.contiguous().cpu() for k, v in state.items()}
    save_file(clean, str(out_path))

    metadata = {
        "model_type": "faster-rcnn",
        "version": version,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "metrics": {
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map5095,
        },
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
    parser.add_argument("--map5095", type=float, default=None)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=_PROJECT_ROOT / "registry",
    )
    args = parser.parse_args(argv)

    out = export_and_promote(
        checkpoint=args.checkpoint,
        version=args.version,
        dataset_version=args.dataset_version,
        dataset_hash=args.dataset_hash,
        map50=args.map50,
        map5095=args.map5095,
        registry_root=args.registry_root,
    )
    print(f"Promoted Faster R-CNN to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
