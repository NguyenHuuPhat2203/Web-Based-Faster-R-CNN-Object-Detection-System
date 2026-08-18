#!/usr/bin/env python3
"""Export a ``.pth`` checkpoint to ``.safetensors`` and promote to the registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.scripts.registry import RegistryGuardError, write_registry_entry

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _convert_to_safetensors(src: Path, dest: Path) -> None:
    """Load a torch checkpoint and save it as safetensors."""
    state = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # safetensors requires contiguous CPU tensors
    clean = {k: v.contiguous().cpu() for k, v in state.items()}
    save_file(clean, str(dest))


def export_and_promote(
    *,
    checkpoint: Path,
    version: str,
    dataset_version: str,
    dataset_hash: str,
    map50: float | None,
    map5095: float | None,
    registry_root: Path,
    i_know: bool = False,
) -> Path:
    """Convert a checkpoint to safetensors and write a registry entry.

    Returns:
        Path to the promoted ``model.safetensors``.
    """
    return write_registry_entry(
        model_key="faster_rcnn",
        version=version,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        metrics={"mAP@0.5": map50, "mAP@0.5:0.95": map5095},
        weights_src=checkpoint,
        weights_name="model.safetensors",
        convert=_convert_to_safetensors,
        registry_root=registry_root,
        i_know=i_know,
        source_checkpoint=str(checkpoint),
    )


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
    parser.add_argument("--i-know", action="store_true", help="Allow v1.0.0")
    args = parser.parse_args(argv)

    try:
        out = export_and_promote(
            checkpoint=args.checkpoint,
            version=args.version,
            dataset_version=args.dataset_version,
            dataset_hash=args.dataset_hash,
            map50=args.map50,
            map5095=args.map5095,
            registry_root=args.registry_root,
            i_know=args.i_know,
        )
    except RegistryGuardError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Promoted Faster R-CNN to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
