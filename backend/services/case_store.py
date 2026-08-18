"""BraTS case store: I/O, parse, and cache behind the case-viewer seam.

One module owns the on-disk layout of a BraTS case directory plus both caches
(parsed cases and predicted masks). The router and the segmentation
orchestration never glob NIfTI files themselves — they ask the store.

Region vocabulary (labels/names/colors) comes from ``ml.core.labels``.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from ml.core.labels import BY_LABEL

from ..config import get_settings

_SEG_SUFFIX = "_seg.nii.gz"
_BASE_SUFFIX = "_t1ce.nii.gz"
_VOLUME_KINDS = ("t1ce", "seg")


@dataclass
class _ParsedCase:
    seg: np.ndarray
    spacing: tuple[float, float, float]
    slices: int


@dataclass
class PredictedMask:
    """A model-produced mask held in the store until served."""

    mask: np.ndarray
    affine: np.ndarray
    seconds: float


_parsed: dict[str, _ParsedCase] = {}
_predicted: dict[str, PredictedMask] = {}


def reset_case_store_cache() -> None:
    """Clear both in-memory caches (tests / invalidation)."""
    _parsed.clear()
    _predicted.clear()


def _case_dir(case_id: str) -> Path:
    """Validated case directory; raises FileNotFoundError when absent."""
    case_dir = Path(get_settings().case_viewer_dir) / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case {case_id} not found")
    return case_dir


def _voxel_volume_cm3(spacing: tuple[float, float, float]) -> float:
    """Voxel volume in cm3 from (x, y, z) spacing in mm."""
    return spacing[0] * spacing[1] * spacing[2] / 1000.0


def _find_case_files(case_dir: Path) -> tuple[Path, Path]:
    """Return (seg, base) NIfTI paths for a case directory."""
    segs = sorted(case_dir.glob(f"*{_SEG_SUFFIX}"))
    bases = sorted(case_dir.glob(f"*{_BASE_SUFFIX}"))
    if not segs:
        raise FileNotFoundError(f"no {_SEG_SUFFIX} in {case_dir}")
    if not bases:
        raise FileNotFoundError(f"no {_BASE_SUFFIX} in {case_dir}")
    return segs[0], bases[0]


def _load_case(case_id: str) -> _ParsedCase:
    """Load and cache a parsed case volume."""
    if case_id not in _parsed:
        seg_path, _ = _find_case_files(_case_dir(case_id))
        seg_img = nib.load(seg_path)
        _parsed[case_id] = _ParsedCase(
            seg=seg_img.get_fdata().astype(np.uint8),
            spacing=tuple(float(z) for z in seg_img.header.get_zooms()[:3]),
            slices=seg_img.shape[2],
        )
    return _parsed[case_id]


def case_dir(case_id: str) -> Path:
    """Validated case directory for model I/O; raises FileNotFoundError."""
    return _case_dir(case_id)


def analyze(case_id: str) -> dict:
    """Per-region tumor volumes (cm3) and metadata for a case."""
    case = _load_case(case_id)
    voxel_cm3 = _voxel_volume_cm3(case.spacing)
    volumes = {
        region.name.lower(): round(int((case.seg == region.label).sum()) * voxel_cm3, 3)
        for region in BY_LABEL.values()
    }
    volumes["total"] = round(sum(volumes.values()), 3)
    return {"case_id": case_id, "slices": case.slices, "volumes_cm3": volumes}


def list_cases() -> list[dict]:
    """All analyzable cases, largest tumor volume first."""
    root = Path(get_settings().case_viewer_dir)
    cases = []
    for case_d in root.iterdir():
        if not case_d.is_dir():
            continue
        if not list(case_d.glob(f"*{_SEG_SUFFIX}")):
            continue
        try:
            cases.append(analyze(case_d.name))
        except Exception:
            continue
    cases.sort(key=lambda c: c["volumes_cm3"]["total"], reverse=True)
    return cases


def volume_path(case_id: str, kind: str) -> Path:
    """Path of a source volume (kind: t1ce | seg) for streaming."""
    if kind not in _VOLUME_KINDS:
        raise ValueError(f"unknown volume kind {kind}")
    matches = sorted(_case_dir(case_id).glob(f"*_{kind}.nii*"))
    if not matches:
        raise FileNotFoundError(f"no {kind} volume for {case_id}")
    return matches[0]


def _slice_to_png(array: np.ndarray) -> str:
    """Encode a 2D array as a base64 grayscale PNG."""
    img = Image.fromarray(array.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_png(mask_slice: np.ndarray) -> str:
    """Encode a label slice as a base64 RGBA PNG (transparent background)."""
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for region in BY_LABEL.values():
        sel = mask_slice == region.label
        rgba[sel, 0] = region.color[0]
        rgba[sel, 1] = region.color[1]
        rgba[sel, 2] = region.color[2]
        rgba[sel, 3] = 220
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def slice_payload(case_id: str, z: int) -> dict:
    """One slice as base64 PNGs (base T1-CE image + region mask)."""
    case = _load_case(case_id)
    z = max(0, min(z, case.slices - 1))

    _, base_path = _find_case_files(_case_dir(case_id))
    base_img = nib.load(base_path)
    base_slice = np.asarray(base_img.dataobj[..., z])
    lo, hi = float(np.percentile(base_slice, 1)), float(np.percentile(base_slice, 99))
    base_slice = np.clip((base_slice - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)

    return {
        "slice_index": z,
        "slices": case.slices,
        "base": _slice_to_png(base_slice),
        "mask": _mask_to_png(case.seg[..., z]),
    }


def put_mask(
    case_id: str, mask: np.ndarray, affine: np.ndarray, seconds: float
) -> None:
    """Store a predicted mask (called by the segmentation orchestration)."""
    _predicted[case_id] = PredictedMask(mask=mask, affine=affine, seconds=seconds)


def get_mask(case_id: str) -> PredictedMask | None:
    """The stored predicted mask, or None before segmentation runs."""
    return _predicted.get(case_id)


def mask_bytes(case_id: str) -> bytes | None:
    """NIfTI bytes of the predicted mask, or None when not segmented."""
    entry = _predicted.get(case_id)
    if entry is None:
        return None
    return nib.Nifti1Image(entry.mask, entry.affine).to_bytes()
