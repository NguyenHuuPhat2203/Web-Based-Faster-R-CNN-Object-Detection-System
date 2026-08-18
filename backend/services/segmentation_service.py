"""Live 3D segmentation orchestration (SwinUNETR BRATS21 pretrained).

Runs the model pipeline and hands results to the case store; holds no cache
of its own. Unwired from the router while the smaller model trains on Colab —
the store seam (``put_mask`` / ``mask_bytes``) is already in place.
"""

from __future__ import annotations

from ml.backbone.segment3d import SwinUNETRSegmentor

from . import case_store


def segment_case(case_id: str) -> dict:
    """Segment a case once; returns metrics (cached in the store on repeats)."""
    case_dir = case_store.case_dir(case_id)
    entry = case_store.get_mask(case_id)
    if entry is None:
        mask, affine, seconds = SwinUNETRSegmentor.get().segment_volume(case_dir)
        case_store.put_mask(case_id, mask, affine, seconds)
        entry = case_store.get_mask(case_id)
    assert entry is not None
    vols = SwinUNETRSegmentor.get().volumes_cm3(entry.mask, entry.affine)
    return {
        "case_id": case_id,
        "seconds": round(entry.seconds, 1),
        "volumes_cm3": {k: round(v, 1) for k, v in vols.items()},
    }


def predicted_mask_bytes(case_id: str) -> bytes:
    """NIfTI bytes of the predicted mask (uncompressed, matches /volume loads)."""
    data = case_store.mask_bytes(case_id)
    if data is None:
        raise KeyError(case_id)
    return data
