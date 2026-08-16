"""YOLOv8 inference pipeline."""

from __future__ import annotations

import io
import time

import structlog
import torch
from PIL import Image as PILImage

from ml.backbone.yolo import get_class_names, load_yolo, reset_model_cache
from ml.core.constants import YOLO_MODEL_VERSION
from ml.core.entities import InferenceResult

from ..config import get_settings
from .memory import free_gpu_memory, is_oom
from .timing import stage_timer

logger = structlog.get_logger(__name__)


def reset_pipeline_cache() -> None:
    """Clear YOLO model cache (tests)."""
    reset_model_cache()


def run_yolo(
    contents: bytes,
    *,
    threshold: float = 0.5,
    min_score: float | None = None,
) -> tuple[PILImage.Image, InferenceResult]:
    """Run YOLOv8 inference on raw image bytes.

    Args:
        contents: Encoded image bytes.
        threshold: Confidence threshold (v1 server-side filter).
        min_score: Floor for returned detections; when set, ``threshold`` is NOT
            applied server-side (v2 client-side threshold slider owns filtering)
            and ``min_score`` is the floor instead.

    Returns:
        Tuple of PIL image and ``InferenceResult``.
    """
    with stage_timer("decode", model_version=YOLO_MODEL_VERSION):
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    settings = get_settings()

    load_started = time.perf_counter()
    model = load_yolo(checkpoint_path=settings.yolo_checkpoint)
    structlog.contextvars.bind_contextvars(
        model_load_ms=round((time.perf_counter() - load_started) * 1000, 3)
    )
    class_names = get_class_names()

    conf = threshold if min_score is None else min_score
    try:
        with stage_timer("forward", model_version=YOLO_MODEL_VERSION):
            results = model(pil_image, conf=conf, verbose=False)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not is_oom(exc):
            raise
        logger.warning("cuda_oom_yolo")
        free_gpu_memory()
        try:
            with stage_timer("forward", model_version=YOLO_MODEL_VERSION):
                results = model(pil_image, conf=conf, verbose=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc2:
            if not is_oom(exc2):
                raise
            logger.warning("cpu_fallback", model_type="yolov8")
            free_gpu_memory()
            # Fresh CPU model: not cached, GPU cache untouched (A1).
            model = load_yolo(
                checkpoint_path=settings.yolo_checkpoint, use_cache=False
            ).to("cpu")
            with stage_timer("forward", model_version=YOLO_MODEL_VERSION):
                results = model(pil_image, conf=conf, verbose=False)

    empty = InferenceResult(
        model_type="yolov8",
        model_version=YOLO_MODEL_VERSION,
    )
    if not results:
        logger.info(
            "inference_stage_timing",
            model_type="yolov8",
            model_version=YOLO_MODEL_VERSION,
        )
        return pil_image, empty

    with stage_timer("postprocess", model_version=YOLO_MODEL_VERSION):
        result = results[0]
        boxes_data = result.boxes
        if boxes_data is None or len(boxes_data) == 0:
            logger.info(
                "inference_stage_timing",
                model_type="yolov8",
                model_version=YOLO_MODEL_VERSION,
            )
            return pil_image, empty

        boxes = boxes_data.xyxy.cpu().numpy().tolist()
        scores = boxes_data.conf.cpu().numpy().tolist()
        labels = boxes_data.cls.int().cpu().numpy().tolist()
        label_names = [class_names.get(label, f"Class {label}") for label in labels]

    result = InferenceResult(
        boxes=boxes,
        labels=labels,
        scores=scores,
        label_names=label_names,
        model_type="yolov8",
        model_version=YOLO_MODEL_VERSION,
    )
    logger.info(
        "inference_stage_timing",
        model_type="yolov8",
        model_version=YOLO_MODEL_VERSION,
    )
    return pil_image, result
