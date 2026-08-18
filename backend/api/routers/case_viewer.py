"""Case-viewer routes (ticket 22): BraTS sample cases with multi-region masks."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ...models import User
from ...services import case_store
from ..dependencies import get_current_user

router = APIRouter(prefix="/case-viewer", tags=["case-viewer"])


@router.get("/cases/{case_id}/volume")
def get_volume(
    case_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    type: str = "t1ce",
) -> FileResponse:
    """Stream a NIfTI volume (t1ce|seg) for client-side 3D rendering."""
    try:
        path = case_store.volume_path(case_id, type)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{case_id}_{type}.nii.gz",
    )


@router.get("/cases")
def list_cases(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """List sample cases with per-region tumor volumes (cm3), largest first."""
    return {"cases": case_store.list_cases()}


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
        raise HTTPException(status_code=404, detail=str(exc)) from exc
