"""Case-viewer routes: BraTS sample cases with multi-region masks."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...models import User
from ...services import case_store, segmentation_service
from ..dependencies import get_current_user

router = APIRouter(prefix="/case-viewer", tags=["case-viewer"])

_CASE_NOT_FOUND = "case not found"


@router.get("/cases")
def list_cases(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """List sample cases with per-region tumor volumes (cm3), largest first."""
    return {"cases": case_store.list_cases()}


@router.get("/cases/{case_id}/volume", response_model=None)
def get_volume(
    case_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    type: Literal["t1ce", "seg", "pred"] = "t1ce",
) -> FileResponse | Response:
    """Stream a NIfTI volume for client-side 3D rendering.

    ``pred`` is the model mask (uncompressed ``.nii``) and exists only after
    ``POST /cases/{case_id}/segment``.
    """
    if type == "pred":
        data = case_store.mask_bytes(case_id)
        if data is None:
            raise HTTPException(status_code=404, detail="case not segmented")
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{case_id}_pred.nii"'
            },
        )
    try:
        path = case_store.volume_path(case_id, type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_CASE_NOT_FOUND) from exc
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{case_id}_{type}.nii.gz",
    )


@router.get("/cases/{case_id}/slices/{z}")
def get_slice(
    case_id: str,
    z: int,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return one slice as base64 PNGs (base T1-CE image + 3-region mask)."""
    try:
        return case_store.slice_payload(case_id, z=z)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_CASE_NOT_FOUND) from exc


@router.post("/cases/{case_id}/segment")
def segment_case(
    case_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Run SwinUNETR 3D segmentation once per case; returns region volumes."""
    try:
        return segmentation_service.segment_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_CASE_NOT_FOUND) from exc
    except segmentation_service.SegmentationUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail="segmentation model unavailable"
        ) from exc
