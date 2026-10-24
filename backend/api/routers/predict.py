"""Prediction routes for Faster R-CNN and YOLOv8.

v1 (``/predict``, ``/predict-yolo``) returns the bare ``PredictionResult`` and
filters at the requested threshold. v2 (``/v2/predict``, ``/v2/predict/yolo``)
returns an item-schema envelope and keeps every detection at or above
``min(threshold, V2_SCORE_FLOOR)`` so the client slider can filter locally.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ml.core.constants import DEFAULT_SCORE_THRESHOLD

from ...core.database import get_db
from ...core.exceptions import DomainError
from ...models import User
from ...pipeline.adapters import DEFAULT_ADAPTERS
from ...schemas.prediction import (
    InferenceItemError,
    InferenceItemOk,
    ItemError,
    PredictionResult,
)
from ...services.file_validation import read_image_upload
from ...services.prediction_service import PredictionService
from ..dependencies import get_current_user

router = APIRouter(tags=["detection"])
logger = structlog.get_logger(__name__)

V2_SCORE_FLOOR = 0.1

UploadArg = Annotated[UploadFile, File(...)]
UserArg = Annotated[User, Depends(get_current_user)]
DbArg = Annotated[Session, Depends(get_db)]
ThresholdArg = Annotated[
    float, Query(ge=0.0, le=1.0, description="Client confidence threshold")
]


async def _run(
    *,
    model_type: str,
    file: UploadFile,
    user: User,
    db: Session,
    score_floor: float,
    threshold: float,
    explain: bool = False,
) -> PredictionResult:
    """Validate the upload, then run blocking inference off the event loop."""
    contents = await read_image_upload(file)
    return await run_in_threadpool(
        PredictionService(db, DEFAULT_ADAPTERS).run,
        model_type=model_type,
        user=user,
        contents=contents,
        original_name=file.filename,
        score_floor=score_floor,
        threshold=threshold,
        explain=explain,
    )


async def _run_v2(**kwargs) -> InferenceItemOk | InferenceItemError:
    """Wrap ``_run`` in the v2 item envelope; domain errors stay HTTP errors."""
    try:
        return InferenceItemOk(result=await _run(**kwargs))
    except DomainError:
        raise
    except Exception:
        logger.exception("inference_failed", model_type=kwargs["model_type"])
        return InferenceItemError(
            error=ItemError(code="INFERENCE_FAILED", message="Inference failed")
        )


@router.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadArg,
    current_user: UserArg,
    db: DbArg,
    threshold: ThresholdArg = DEFAULT_SCORE_THRESHOLD,
) -> PredictionResult:
    """Run Faster R-CNN (with Grad-CAM) on an uploaded MRI image."""
    return await _run(
        model_type="faster-rcnn",
        file=file,
        user=current_user,
        db=db,
        score_floor=threshold,
        threshold=threshold,
        explain=True,
    )


@router.post("/predict-yolo", response_model=PredictionResult)
async def predict_yolo(
    file: UploadArg,
    current_user: UserArg,
    db: DbArg,
    threshold: ThresholdArg = DEFAULT_SCORE_THRESHOLD,
) -> PredictionResult:
    """Run YOLOv8 on an uploaded MRI image."""
    return await _run(
        model_type="yolov8",
        file=file,
        user=current_user,
        db=db,
        score_floor=threshold,
        threshold=threshold,
    )


@router.post("/v2/predict", tags=["detection-v2"])
async def predict_v2(
    file: UploadArg,
    current_user: UserArg,
    db: DbArg,
    threshold: ThresholdArg = DEFAULT_SCORE_THRESHOLD,
    explain: Annotated[
        bool, Query(description="When true, compute Grad-CAM heatmap")
    ] = False,
) -> InferenceItemOk | InferenceItemError:
    """Run Faster R-CNN; returns a single item-schema object."""
    return await _run_v2(
        model_type="faster-rcnn",
        file=file,
        user=current_user,
        db=db,
        score_floor=min(threshold, V2_SCORE_FLOOR),
        threshold=threshold,
        explain=explain,
    )


@router.post("/v2/predict/yolo", tags=["detection-v2"])
async def predict_yolo_v2(
    file: UploadArg,
    current_user: UserArg,
    db: DbArg,
    threshold: ThresholdArg = DEFAULT_SCORE_THRESHOLD,
) -> InferenceItemOk | InferenceItemError:
    """Run YOLOv8; returns a single item-schema object."""
    return await _run_v2(
        model_type="yolov8",
        file=file,
        user=current_user,
        db=db,
        score_floor=min(threshold, V2_SCORE_FLOOR),
        threshold=threshold,
    )
