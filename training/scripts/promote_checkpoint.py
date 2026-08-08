#!/usr/bin/env python3
"""Promote a trained checkpoint into the model registry with an explicit version.

Refuses to write ``v1.0.0`` unless ``--i-know`` is passed, to avoid overwriting
production registry entries with smoke weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from training.scripts.export_safetensors import export_and_promote
from training.scripts.promote_yolo import promote_yolo


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("faster_rcnn", "yolo"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--version",
        required=True,
        help="Registry version directory name (e.g. v0.0.0-smoke or v1.1.0)",
    )
    parser.add_argument(
        "--i-know",
        action="store_true",
        help="Required when --version is v1.0.0 (production slot)",
    )
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-hash", default="")
    parser.add_argument("--map50", type=float, default=None)
    parser.add_argument("--map5095", type=float, default=None)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=_PROJECT_ROOT / "registry",
    )
    args = parser.parse_args(argv)

    if args.version == "v1.0.0" and not args.i_know:
        print(
            "Refusing to promote to v1.0.0 without --i-know "
            "(use a smoke version like v0.0.0-smoke, or pass --i-know).",
            file=sys.stderr,
        )
        return 2

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    if args.model == "faster_rcnn":
        out = export_and_promote(
            checkpoint=args.checkpoint,
            version=args.version,
            dataset_version=args.dataset_version,
            dataset_hash=args.dataset_hash,
            map50=args.map50,
            map5095=args.map5095,
            registry_root=args.registry_root,
        )
    else:
        out = promote_yolo(
            checkpoint=args.checkpoint,
            version=args.version,
            dataset_version=args.dataset_version,
            dataset_hash=args.dataset_hash,
            map50=args.map50,
            registry_root=args.registry_root,
        )

    print(f"Promoted {args.model} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
