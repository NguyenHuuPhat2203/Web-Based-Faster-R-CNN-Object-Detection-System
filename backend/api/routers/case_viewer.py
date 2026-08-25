"""Case-viewer routes (ticket 22): BraTS sample cases with multi-region masks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...config import get_settings
from ...models import User
from ...services import case_viewer_service, segmentation_service
from ..dependencies import get_current_user

router = APIRouter(prefix="/case-viewer", tags=["case-viewer"])


@router.post("/cases/{case_id}/segment")
def segment_case(
    case_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Run live 3D segmentation (SwinUNETR) on a case; returns region metrics."""
    try:
        return segmentation_service.segment_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/volume")
def get_volume(
    case_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    type: str = "t1ce",
) -> Response:
    """Stream a NIfTI volume (t1ce|seg|pred) for client-side 3D rendering."""
    if type == "pred":
        try:
            data = segmentation_service.predicted_mask_bytes(case_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="case not segmented yet"
            ) from exc
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"content-disposition": f"attachment; filename={case_id}_pred.nii"},
        )
    settings = get_settings()
    case_dir = Path(settings.case_viewer_dir) / case_id
    matches = sorted(case_dir.glob(f"*_{type}.nii*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"no {type} volume for {case_id}")
    return FileResponse(
        matches[0],
        media_type="application/octet-stream",
        filename=f"{case_id}_{type}.nii.gz",
    )


@router.get("/cases")
def list_cases(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """List sample cases with per-region tumor volumes (cm3), largest first."""
    settings = get_settings()
    return {"cases": case_viewer_service.list_cases(str(settings.case_viewer_dir))}


@router.get("/cases/{case_id}/slices/{z}")
def get_slice(
    case_id: str,
    z: int,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return one slice as base64 PNGs (base T1-CE image + 3-region mask)."""
    settings = get_settings()
    case_dir = str(Path(settings.case_viewer_dir) / case_id)
    try:
        return case_viewer_service.slice_payload(case_dir, z=z)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
