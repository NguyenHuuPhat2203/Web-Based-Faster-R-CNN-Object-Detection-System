"""Faster R-CNN inference pipeline."""

from __future__ import annotations

import io
import time

import structlog
import torch
from PIL import Image as PILImage

from ml.backbone.faster_rcnn import get_device, load_faster_rcnn, reset_model_cache
from ml.core.constants import FRCNN_CLASS_NAMES, FRCNN_MODEL_VERSION
from ml.core.entities import InferenceResult
from ml.core.postprocess import filter_by_threshold, tensors_to_inference_result
from ml.transforms import image_to_tensor

from ..config import get_settings
from .gradcam import GradCAM
from .memory import free_gpu_memory
from .timing import stage_timer

logger = structlog.get_logger(__name__)

_gradcam: GradCAM | None = None


def _load_model_and_gradcam() -> tuple[torch.nn.Module, GradCAM]:
    """Load Faster R-CNN and attach a Grad-CAM helper."""
    global _gradcam
    settings = get_settings()
    model = load_faster_rcnn(
        checkpoint_path=settings.faster_rcnn_checkpoint,
        safetensors_path=settings.faster_rcnn_safetensors,
        hf_repo=settings.hf_checkpoint_repo,
        hf_filename=settings.hf_checkpoint_filename,
        hf_token=settings.hf_access_token or None,
        device=get_device(),
    )
    if _gradcam is None or _gradcam.model is not model:
        _gradcam = GradCAM(model)
    return model, _gradcam


def reset_pipeline_cache() -> None:
    """Clear model and Grad-CAM caches (tests)."""
    global _gradcam
    if _gradcam is not None:
        _gradcam.remove_hooks()
        _gradcam = None
    reset_model_cache()


@torch.inference_mode()
def _forward(model: torch.nn.Module, image_tensor: torch.Tensor) -> list[dict]:
    """Run a single-image forward pass."""
    return model([image_tensor])


def run_faster_rcnn(
    contents: bytes,
    *,
    threshold: float = 0.5,
    explain: bool = True,
) -> tuple[PILImage.Image, InferenceResult]:
    """Run Faster R-CNN inference on raw image bytes.

    Args:
        contents: Encoded image bytes.
        threshold: Confidence threshold.
        explain: When True, compute Grad-CAM if detections exist (v1 default).

    Returns:
        Tuple of PIL image and ``InferenceResult``.
    """
    with stage_timer("decode", model_version=FRCNN_MODEL_VERSION):
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    device = get_device()
    with stage_timer("tensor", model_version=FRCNN_MODEL_VERSION):
        image_tensor = image_to_tensor(pil_image, device=device)

    load_started = time.perf_counter()
    model, gradcam = _load_model_and_gradcam()
    structlog.contextvars.bind_contextvars(
        model_load_ms=round((time.perf_counter() - load_started) * 1000, 3)
    )

    try:
        with stage_timer("forward", model_version=FRCNN_MODEL_VERSION):
            predictions = _forward(model, image_tensor)
    except torch.cuda.OutOfMemoryError:
        logger.warning("cuda_oom_frcnn", input_shape=list(image_tensor.shape))
        free_gpu_memory()
        try:
            with stage_timer("forward", model_version=FRCNN_MODEL_VERSION):
                predictions = _forward(model, image_tensor)
        except torch.cuda.OutOfMemoryError:
            free_gpu_memory()
            raise

    with stage_timer("postprocess", model_version=FRCNN_MODEL_VERSION):
        prediction = predictions[0]
        boxes, scores, labels = filter_by_threshold(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            threshold=threshold,
        )
        label_list = (
            [int(x) for x in labels.detach().cpu().tolist()] if labels.numel() else []
        )

    heatmap: str | None = None
    structlog.contextvars.bind_contextvars(gradcam_ms=0.0)
    if explain and label_list:
        try:
            with (
                torch.enable_grad(),
                stage_timer("gradcam", model_version=FRCNN_MODEL_VERSION),
            ):
                heatmap = gradcam.compute(image_tensor.unsqueeze(0), label_list[0])
        except Exception as exc:
            logger.warning("gradcam_failed", error=str(exc))

    result = tensors_to_inference_result(
        boxes,
        scores,
        labels,
        class_names=FRCNN_CLASS_NAMES,
        model_type="faster-rcnn",
        model_version=FRCNN_MODEL_VERSION,
        heatmap=heatmap,
    )
    logger.info(
        "inference_stage_timing",
        model_type="faster-rcnn",
        model_version=FRCNN_MODEL_VERSION,
    )
    return pil_image, result
