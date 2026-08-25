"""Case-viewer service: analyze BraTS NIfTI volumes into region masks + cm3 metrics.

Demo-first (ticket 22): masks come from the data's existing ``*_seg.nii.gz``
(expert-drawn), parsed server-side with nibabel. Region labels: 1=NCR (necrosis),
2=ED (edema), 4=ET (enhancing tumor).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

# Region colors (RGBA overlay) — ET red, ED green, NCR amber.
REGION_COLORS: dict[int, tuple[int, int, int]] = {
    1: (245, 158, 11),  # NCR — amber
    2: (16, 185, 129),  # ED — emerald
    4: (239, 68, 68),  # ET — red
}
REGION_NAMES: dict[int, str] = {1: "NCR", 2: "ED", 4: "ET"}

_SEG_SUFFIX = "_seg.nii.gz"
_BASE_SUFFIX = "_t1ce.nii.gz"


@dataclass
class _ParsedCase:
    seg: np.ndarray
    spacing: tuple[float, float, float]
    slices: int


_cache: dict[str, _ParsedCase] = {}


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


def _load_case(case_dir: str) -> _ParsedCase:
    """Load and cache a parsed case volume."""
    key = str(case_dir)
    if key not in _cache:
        seg_path, _ = _find_case_files(Path(case_dir))
        seg_img = nib.load(seg_path)
        _cache[key] = _ParsedCase(
            seg=seg_img.get_fdata().astype(np.uint8),
            spacing=tuple(float(z) for z in seg_img.header.get_zooms()[:3]),
            slices=seg_img.shape[2],
        )
    return _cache[key]


def analyze_case(case_dir: str) -> dict:
    """Compute per-region tumor volumes (cm3) and metadata for a case."""
    case = _load_case(case_dir)
    voxel_cm3 = _voxel_volume_cm3(case.spacing)
    volumes = {
        REGION_NAMES[label].lower(): round(
            int((case.seg == label).sum()) * voxel_cm3, 3
        )
        for label in (1, 2, 4)
    }
    volumes["total"] = round(sum(volumes.values()), 3)
    return {
        "case_id": Path(case_dir).name,
        "path": str(case_dir),
        "slices": case.slices,
        "volumes_cm3": volumes,
    }


def list_cases(root_dir: str) -> list[dict]:
    """List all analyzable cases, largest tumor volume first."""
    root = Path(root_dir)
    cases = []
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        if not list(case_dir.glob(f"*{_SEG_SUFFIX}")):
            continue
        try:
            cases.append(analyze_case(str(case_dir)))
        except Exception:
            continue
    cases.sort(key=lambda c: c["volumes_cm3"]["total"], reverse=True)
    return cases


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
    for label, (r, g, b) in REGION_COLORS.items():
        sel = mask_slice == label
        rgba[sel, 0] = r
        rgba[sel, 1] = g
        rgba[sel, 2] = b
        rgba[sel, 3] = 220
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def slice_payload(case_dir: str, z: int) -> dict:
    """Return one slice as base64 PNGs (base T1-CE image + 3-region mask)."""
    case = _load_case(case_dir)
    z = max(0, min(z, case.slices - 1))

    _, base_path = _find_case_files(Path(case_dir))
    base_img = nib.load(base_path)
    base_slice = np.asarray(base_img.dataobj[..., z])
    # normalize to 8-bit for display
    lo, hi = float(np.percentile(base_slice, 1)), float(np.percentile(base_slice, 99))
    base_slice = np.clip((base_slice - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)

    return {
        "slice_index": z,
        "slices": case.slices,
        "base": _slice_to_png(base_slice),
        "mask": _mask_to_png(case.seg[..., z]),
    }


def reset_case_viewer_cache() -> None:
    """Clear the parsed-case cache (tests)."""
    _cache.clear()
