"""Deep inference module: decode → fallback → floor filter → result."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog
import torch
from PIL import Image as PILImage
from PIL import ImageOps

from ml.core.entities import InferenceResult
from ml.core.postprocess import filter_above_floor, tensors_to_inference_result
from ml.models.faster_rcnn import get_device

from .memory import free_gpu_memory, is_oom
from .timing import stage_timer

logger = structlog.get_logger(__name__)


@dataclass
class RawDetections:
    """Model output in a common shape: CPU tensors in xyxy format."""

    boxes: torch.Tensor  # [N, 4]
    scores: torch.Tensor  # [N]
    labels: torch.Tensor  # [N] int


class ModelAdapter(ABC):
    """Seam: per-model preprocessing, forward pass, and optional explainability."""

    model_type: str = ""
    model_version: str = ""
    class_names: dict[int, str] = {}

    @abstractmethod
    def forward(self, image: PILImage.Image, device: torch.device) -> RawDetections:
        """Preprocess and run forward on ``device``; detections on CPU."""

    def explain(self, image: PILImage.Image, label: int) -> str | None:
        """Optional Grad-CAM heatmap (base64 PNG) or ``None``."""
        return None


class InferenceEngine:
    """Run a model adapter over bytes with GPU→CPU fallback and floor filtering."""

    def __init__(self, adapter: ModelAdapter) -> None:
        """Wrap a model adapter with decode, fallback, and floor filtering."""
        self.adapter = adapter

    def run(
        self,
        contents: bytes,
        *,
        score_floor: float = 0.0,
        explain: bool = False,
    ) -> InferenceResult:
        """Run inference over image bytes; returns an InferenceResult."""
        with stage_timer("decode", model_version=self.adapter.model_version):
            # Browsers render JPEGs EXIF-rotated; boxes must use the same frame.
            with PILImage.open(io.BytesIO(contents)) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")

        detections, used_cpu = self._forward_with_fallback(image)

        with stage_timer("postprocess", model_version=self.adapter.model_version):
            boxes, scores, labels = filter_above_floor(
                detections.boxes,
                detections.scores,
                detections.labels,
                floor=score_floor,
            )

        heatmap: str | None = None
        if explain and not used_cpu and labels.numel():
            label = int(labels[0].item())
            try:
                with stage_timer("gradcam", model_version=self.adapter.model_version):
                    heatmap = self.adapter.explain(image, label)
            except Exception as exc:
                logger.warning("gradcam_failed", error=str(exc))

        result = tensors_to_inference_result(
            boxes,
            scores,
            labels,
            class_names=self.adapter.class_names,
            model_type=self.adapter.model_type,
            model_version=self.adapter.model_version,
            heatmap=heatmap,
        )
        logger.info(
            "inference_stage_timing",
            model_type=self.adapter.model_type,
            model_version=self.adapter.model_version,
        )
        return result

    def _forward_with_fallback(
        self, image: PILImage.Image
    ) -> tuple[RawDetections, bool]:
        device = get_device()
        try:
            return self._forward(image, device), False
        except Exception as exc:
            if not is_oom(exc):
                raise
            logger.warning("cuda_oom", model_type=self.adapter.model_type)
            free_gpu_memory()
            try:
                return self._forward(image, device), False
            except Exception as exc2:
                if not is_oom(exc2):
                    raise
                logger.warning("cpu_fallback", model_type=self.adapter.model_type)
                free_gpu_memory()
                return self._forward(image, torch.device("cpu")), True

    def _forward(self, image: PILImage.Image, device: torch.device) -> RawDetections:
        with stage_timer("forward", model_version=self.adapter.model_version):
            return self.adapter.forward(image, device)
