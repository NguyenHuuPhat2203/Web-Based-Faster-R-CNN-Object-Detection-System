"""Faster R-CNN prediction HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...models import User
from ...pipeline.adapters import DEFAULT_ADAPTERS
from ...schemas.prediction import PredictionResult
from ...services.file_validation import validate_image_upload
from ...services.prediction_service import PredictionService
from ..dependencies import get_current_user

router = APIRouter(tags=["detection"])


@router.post("/predict", response_model=PredictionResult)
async def predict(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    threshold: float = 0.5,
) -> PredictionResult:
    """Run Faster R-CNN inference on an uploaded MRI image."""
    contents = await file.read()
    validate_image_upload(content_type=file.content_type, size_bytes=len(contents))
    return PredictionService(db, DEFAULT_ADAPTERS).run(
        model_type="faster-rcnn",
        user=current_user,
        contents=contents,
        original_name=file.filename,
        content_type=file.content_type,
        score_floor=threshold,
        threshold=threshold,
        explain=True,
    )
