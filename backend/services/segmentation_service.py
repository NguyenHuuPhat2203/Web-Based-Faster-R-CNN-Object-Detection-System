"""Live 3D segmentation service (SwinUNETR BRATS21 pretrained).

Wraps ``ml.backbone.segment3d`` with a per-case in-memory cache so the
~2-minute GPU inference runs once per case and the predicted mask is
streamable for NiiVue overlay.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ml.backbone.segment3d import SwinUNETRSegmentor

from ..config import get_settings

# case_id -> (mask uint8 (D,H,W), affine, seconds)
_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}


def reset_segmentation_cache() -> None:
    """Clear the in-memory predicted-mask cache (tests / cache invalidation)."""
    _cache.clear()


def segment_case(case_id: str) -> dict:
    """Segment a case once; returns metrics (cached on repeat calls)."""
    if case_id not in _cache:
        case_dir = Path(get_settings().case_viewer_dir) / case_id
        if not case_dir.is_dir():
            raise ValueError(f"case {case_id} not found")
        mask, affine, seconds = SwinUNETRSegmentor.get().segment_volume(case_dir)
        _cache[case_id] = (mask, affine, seconds)
    mask, affine, seconds = _cache[case_id]
    vols = SwinUNETRSegmentor.get().volumes_cm3(mask, affine)
    return {
        "case_id": case_id,
        "seconds": round(seconds, 1),
        "volumes_cm3": {k: round(v, 1) for k, v in vols.items()},
    }


def predicted_mask_bytes(case_id: str) -> bytes:
    """NIfTI bytes of the predicted mask (uncompressed, matches /volume loads)."""
    if case_id not in _cache:
        raise KeyError(case_id)
    mask, affine, _ = _cache[case_id]
    return nib.Nifti1Image(mask, affine).to_bytes()
