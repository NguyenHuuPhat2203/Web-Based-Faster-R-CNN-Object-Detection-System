"""v2 Faster R-CNN / YOLO prediction routes (item-schema)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...core.exceptions import DomainError
from ...models import User
from ...pipeline.adapters import DEFAULT_ADAPTERS
from ...schemas.prediction import InferenceItemError, InferenceItemOk, ItemError
from ...services.file_validation import validate_image_upload
from ...services.prediction_service import PredictionService
from ..dependencies import get_current_user

router = APIRouter(prefix="/v2", tags=["detection-v2"])


@router.post("/predict")
async def predict_v2(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    threshold: float = 0.5,
    explain: Annotated[
        bool,
        Query(description="When true, compute Grad-CAM heatmap"),
    ] = False,
) -> InferenceItemOk | InferenceItemError:
    """Run Faster R-CNN inference; returns a single item-schema object."""
    contents = await file.read()
    validate_image_upload(content_type=file.content_type, size_bytes=len(contents))
    try:
        result = PredictionService(db, DEFAULT_ADAPTERS).run(
            model_type="faster-rcnn",
            user=current_user,
            contents=contents,
            original_name=file.filename,
            content_type=file.content_type,
            score_floor=0.1,
            threshold=threshold,
            explain=explain,
        )
        return InferenceItemOk(result=result)
    except DomainError:
        raise
    except Exception as exc:
        return InferenceItemError(
            error=ItemError(code="INFERENCE_FAILED", message=str(exc))
        )


@router.post("/predict/yolo")
async def predict_yolo_v2(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    threshold: float = 0.5,
) -> InferenceItemOk | InferenceItemError:
    """Run YOLOv8 inference; returns a single item-schema object."""
    contents = await file.read()
    validate_image_upload(content_type=file.content_type, size_bytes=len(contents))
    try:
        result = PredictionService(db, DEFAULT_ADAPTERS).run(
            model_type="yolov8",
            user=current_user,
            contents=contents,
            original_name=file.filename,
            content_type=file.content_type,
            score_floor=0.1,
            threshold=threshold,
        )
        return InferenceItemOk(result=result)
    except DomainError:
        raise
    except Exception as exc:
        return InferenceItemError(
            error=ItemError(code="INFERENCE_FAILED", message=str(exc))
        )
