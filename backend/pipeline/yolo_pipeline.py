"""YOLOv8 inference pipeline."""

from __future__ import annotations

import io
import time

import structlog
from PIL import Image as PILImage

from ml.backbone.yolo import get_class_names, load_yolo, reset_model_cache
from ml.core.constants import YOLO_MODEL_VERSION
from ml.core.entities import InferenceResult

from ..config import get_settings
from .memory import free_gpu_memory
from .timing import stage_timer

logger = structlog.get_logger(__name__)


def reset_pipeline_cache() -> None:
    """Clear YOLO model cache (tests)."""
    reset_model_cache()


def run_yolo(
    contents: bytes,
    *,
    threshold: float = 0.5,
) -> tuple[PILImage.Image, InferenceResult]:
    """Run YOLOv8 inference on raw image bytes.

    Args:
        contents: Encoded image bytes.
        threshold: Confidence threshold.

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

    try:
        with stage_timer("forward", model_version=YOLO_MODEL_VERSION):
            results = model(pil_image, conf=threshold, verbose=False)
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("cuda_oom_yolo")
            free_gpu_memory()
            with stage_timer("forward", model_version=YOLO_MODEL_VERSION):
                results = model(pil_image, conf=threshold, verbose=False)
        else:
            raise

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
