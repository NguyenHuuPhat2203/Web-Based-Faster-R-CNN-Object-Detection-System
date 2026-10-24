"""BraTS case store: I/O, parse, and cache behind the case-viewer seam.

One module owns the on-disk layout of a BraTS case directory plus its caches
(parsed segmentations, per-case summaries, predicted masks). The router and the
segmentation orchestration never glob NIfTI files themselves.

Region vocabulary (labels/names/colors) comes from ``ml.core.labels``.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from PIL import Image

from ml.core.labels import REGIONS

from ..config import get_settings

_SEG_SUFFIX = "_seg.nii.gz"
_BASE_SUFFIX = "_t1ce.nii.gz"
VOLUME_KINDS = ("t1ce", "seg")
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MASK_ALPHA = 220


@dataclass(frozen=True)
class _ParsedCase:
    seg: np.ndarray
    spacing: tuple[float, float, float]
    slices: int


@dataclass(frozen=True)
class PredictedMask:
    """A model-produced mask held in the store until served."""

    mask: np.ndarray
    affine: np.ndarray
    seconds: float


_predicted: dict[str, PredictedMask] = {}


def case_dir(case_id: str) -> Path:
    """Resolve a case directory contained under ``case_viewer_dir``.

    Raises:
        FileNotFoundError: If the id is malformed, escapes the root, or is absent.
    """
    if not _CASE_ID_RE.fullmatch(case_id) or ".." in case_id:
        raise FileNotFoundError("case not found")
    root = Path(get_settings().case_viewer_dir).resolve()
    candidate = (root / case_id).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise FileNotFoundError("case not found")
    return candidate


def _find_case_file(directory: Path, suffix: str) -> Path:
    """Return the first file in ``directory`` ending with ``suffix``."""
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"no {suffix} in {directory.name}")
    return matches[0]


# ponytail: LRU of 8 parsed segmentations (~9 MB each for 240x240x155);
# raise maxsize if the viewer is used with many cases concurrently.
@lru_cache(maxsize=8)
def _load_case(case_id: str) -> _ParsedCase:
    """Load a case's expert segmentation."""
    seg_image = nib.load(_find_case_file(case_dir(case_id), _SEG_SUFFIX))
    zooms = seg_image.header.get_zooms()
    return _ParsedCase(
        seg=np.asarray(seg_image.dataobj, dtype=np.uint8),
        spacing=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
        slices=int(seg_image.shape[2]),
    )


@cache
def analyze(case_id: str) -> dict:
    """Per-region tumor volumes (cm3) and metadata for a case."""
    case = _load_case(case_id)
    voxel_cm3 = case.spacing[0] * case.spacing[1] * case.spacing[2] / 1000.0
    volumes = {
        region.name.lower(): round(
            int(np.count_nonzero(case.seg == region.label)) * voxel_cm3, 3
        )
        for region in REGIONS
    }
    volumes["total"] = round(sum(volumes.values()), 3)
    return {"case_id": case_id, "slices": case.slices, "volumes_cm3": volumes}


def reset_case_store_cache() -> None:
    """Clear every cache (tests / invalidation)."""
    _load_case.cache_clear()
    analyze.cache_clear()
    _predicted.clear()


def list_cases() -> list[dict]:
    """All analyzable cases, largest tumor volume first."""
    root = Path(get_settings().case_viewer_dir)
    if not root.is_dir():
        return []
    cases = []
    for directory in root.iterdir():
        if not directory.is_dir() or not any(directory.glob(f"*{_SEG_SUFFIX}")):
            continue
        try:
            cases.append(analyze(directory.name))
        except (OSError, EOFError, ValueError, ImageFileError):
            continue
    cases.sort(key=lambda case: case["volumes_cm3"]["total"], reverse=True)
    return cases


def volume_path(case_id: str, kind: str) -> Path:
    """Path of a source volume (kind: t1ce | seg) for streaming.

    Raises:
        ValueError: If ``kind`` is unknown.
        FileNotFoundError: If the case or volume is missing.
    """
    if kind not in VOLUME_KINDS:
        raise ValueError(f"unknown volume kind {kind}")
    return _find_case_file(case_dir(case_id), f"_{kind}.nii.gz")


def _encode_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _mask_to_rgba(mask_slice: np.ndarray) -> Image.Image:
    """Colorize a label slice as RGBA (transparent background)."""
    rgba = np.zeros((*mask_slice.shape, 4), dtype=np.uint8)
    for region in REGIONS:
        rgba[mask_slice == region.label] = (*region.color, _MASK_ALPHA)
    return Image.fromarray(rgba, mode="RGBA")


def slice_payload(case_id: str, z: int) -> dict:
    """One axial slice as base64 PNGs (T1-CE base image + region mask).

    ``z`` is clamped into ``[0, slices - 1]``.
    """
    case = _load_case(case_id)
    z = max(0, min(z, case.slices - 1))

    base_image = nib.load(_find_case_file(case_dir(case_id), _BASE_SUFFIX))
    base_slice = np.asarray(base_image.dataobj[..., z], dtype=np.float32)
    low, high = np.percentile(base_slice, (1, 99))
    scaled = np.clip((base_slice - low) / max(high - low, 1e-6) * 255.0, 0, 255)

    return {
        "slice_index": z,
        "slices": case.slices,
        "base": _encode_png(Image.fromarray(scaled.astype(np.uint8), mode="L")),
        "mask": _encode_png(_mask_to_rgba(case.seg[..., z])),
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
    """Uncompressed NIfTI bytes of the predicted mask, or None when absent."""
    entry = _predicted.get(case_id)
    if entry is None:
        return None
    return nib.Nifti1Image(entry.mask, entry.affine).to_bytes()
