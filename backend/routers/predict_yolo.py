"""YOLOv8 prediction route — mirrors predict.py interface but uses Ultralytics YOLOv8."""

from __future__ import annotations

import hashlib
import io
import os
import uuid

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from PIL import Image
from sqlalchemy.orm import Session

from database import get_db
from models import Image as ImageModel, Detection, User
from schemas import PredictionResult
from dependencies import get_current_user
import config as cfg

router = APIRouter(tags=["detection"])

# ── Model initialisation (lazy, singleton) ──────────────────────────

_model = None
_CLASS_NAMES: dict[int, str] = {}


def _load_model():
    global _model, _CLASS_NAMES
    if _model is not None:
        return _model

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "YOLOv8 requires the 'ultralytics' package. "
            "Install it with: pip install ultralytics"
        )

    checkpoint_path = "../checkpoints/yolov8_tumor.pt"
    if os.path.exists(checkpoint_path):
        model = YOLO(checkpoint_path)
    else:
        # Fall back to the pretrained YOLOv8n — will detect COCO classes
        model = YOLO("yolov8n.pt")

    _model = model
    # Build label map from model's built-in class names
    _CLASS_NAMES = {i: name for i, name in model.names.items()}
    return _model


@router.post("/predict-yolo", response_model=PredictionResult)
async def predict_yolo(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ── Validate ──────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    contents = await file.read()
    if len(contents) > cfg.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {cfg.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    # ── Inference ─────────────────────────────────────────────────
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = _load_model()
    # YOLOv8 returns a list of Results objects (one per image)
    results = model(image, conf=threshold, verbose=False)

    if len(results) == 0:
        return PredictionResult(boxes=[], labels=[], scores=[], label_names=[])

    result = results[0]
    boxes_data = result.boxes

    if boxes_data is None or len(boxes_data) == 0:
        return PredictionResult(boxes=[], labels=[], scores=[], label_names=[])

    boxes = boxes_data.xyxy.cpu().numpy().tolist()
    scores = boxes_data.conf.cpu().numpy().tolist()
    labels = boxes_data.cls.int().cpu().numpy().tolist()
    label_names = [_CLASS_NAMES.get(label, f"Class {label}") for label in labels]

    # ── Save to disk + DB ─────────────────────────────────────────
    user_dir = os.path.join(cfg.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, stored_name)

    image.save(file_path)

    db_image = ImageModel(
        user_id=current_user.id,
        original_name=file.filename or "unknown",
        stored_name=stored_name,
        filepath=file_path,
        mime_type=file.content_type or "image/jpeg",
        width=image.width,
        height=image.height,
        file_size=len(contents),
        content_hash=hashlib.sha256(contents).hexdigest(),
        model_type="yolov8",
        model_version="yolov8n",
        threshold=threshold,
    )
    db.add(db_image)
    db.flush()

    for box, label, score, label_name in zip(boxes, labels, scores, label_names):
        db.add(Detection(
            image_id=db_image.id,
            label=label,
            label_name=label_name,
            score=score,
            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        ))
    db.commit()

    return PredictionResult(
        boxes=boxes, labels=labels, scores=scores, label_names=label_names,
        model_type="yolov8",
        model_version="yolov8n",
    )
