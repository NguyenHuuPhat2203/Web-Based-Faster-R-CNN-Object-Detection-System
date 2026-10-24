#!/usr/bin/env python3
"""Promote a YOLOv8 ``.pt`` checkpoint into the model registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.scripts.registry import RegistryGuardError, write_registry_entry

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def promote_yolo(
    *,
    checkpoint: Path,
    version: str,
    dataset_version: str,
    dataset_hash: str,
    map50: float | None,
    registry_root: Path,
    i_know: bool = False,
) -> Path:
    """Copy YOLO weights into the registry and write metadata."""
    return write_registry_entry(
        model_key="yolo",
        version=version,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        metrics={"mAP@0.5": map50},
        weights_src=checkpoint,
        weights_name="yolov8_tumor.pt",
        registry_root=registry_root,
        i_know=i_know,
    )


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
    parser.add_argument("--i-know", action="store_true", help="Allow v1.0.0")
    args = parser.parse_args(argv)

    try:
        out = promote_yolo(
            checkpoint=args.checkpoint,
            version=args.version,
            dataset_version=args.dataset_version,
            dataset_hash=args.dataset_hash,
            map50=args.map50,
            registry_root=args.registry_root,
            i_know=args.i_know,
        )
    except RegistryGuardError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Promoted YOLO to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
