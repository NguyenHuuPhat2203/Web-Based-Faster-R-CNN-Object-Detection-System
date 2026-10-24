"""Live 3D segmentation orchestration (SwinUNETR BRATS21 pretrained).

Runs the model and hands results to the case store; holds no mask cache of its
own.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from ml.models.swin_unetr import SwinUNETRSegmentor, region_volumes_cm3

from ..config import get_settings
from . import case_store

# ponytail: one segmentation at a time; a 128^3 sliding window saturates a
# 4 GB GPU. Swap for a job queue if several users segment concurrently.
_segment_lock = threading.Lock()


class SegmentationUnavailableError(RuntimeError):
    """The segmentation model cannot be loaded (missing or invalid weights)."""


@lru_cache(maxsize=1)
def _segmentor() -> SwinUNETRSegmentor:
    """Load the segmentor once per process."""
    try:
        return SwinUNETRSegmentor(get_settings().swin_unetr_weights)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SegmentationUnavailableError(str(exc)) from exc


def segment_case(case_id: str) -> dict:
    """Segment a case once (repeats reuse the stored mask) and return metrics.

    Raises:
        FileNotFoundError: If the case or one of its modalities is missing.
        SegmentationUnavailableError: If the model cannot be loaded.
    """
    directory = case_store.case_dir(case_id)
    with _segment_lock:
        entry = case_store.get_mask(case_id)
        if entry is None:
            mask, affine, seconds = _segmentor().segment_volume(directory)
            case_store.put_mask(case_id, mask, affine, seconds)
            entry = case_store.PredictedMask(mask, affine, seconds)
    volumes = region_volumes_cm3(entry.mask, entry.affine)
    return {
        "case_id": case_id,
        "seconds": round(entry.seconds, 1),
        "volumes_cm3": {name: round(value, 1) for name, value in volumes.items()},
    }
