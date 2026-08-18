"""Model adapters for Faster R-CNN and YOLO."""

from __future__ import annotations

import torch
from PIL import Image as PILImage

from ml.backbone.faster_rcnn import load_faster_rcnn
from ml.backbone.yolo import load_yolo
from ml.core.constants import (
    FRCNN_CLASS_NAMES,
    FRCNN_MODEL_VERSION,
    YOLO_MODEL_VERSION,
)
from ml.transforms import image_to_tensor

from ..config import get_settings
from .gradcam import GradCAM
from .inference import ModelAdapter, RawDetections


class FasterRCNNAdapter(ModelAdapter):
    """Faster R-CNN: tensor preprocessing + optional Grad-CAM."""

    model_type = "faster-rcnn"
    model_version = FRCNN_MODEL_VERSION
    class_names = FRCNN_CLASS_NAMES

    def __init__(self) -> None:
        """Build the adapter; the model loads lazily on first forward."""
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._gradcam: GradCAM | None = None

    def forward(self, image: PILImage.Image, device: torch.device) -> RawDetections:
        """Preprocess and run Faster R-CNN forward on ``device``."""
        settings = get_settings()
        # A1: cache only the GPU model; a fresh CPU model leaves the cache intact.
        self._model = load_faster_rcnn(
            checkpoint_path=settings.faster_rcnn_checkpoint,
            safetensors_path=settings.faster_rcnn_safetensors,
            hf_repo=settings.hf_checkpoint_repo,
            hf_filename=settings.hf_checkpoint_filename,
            hf_token=settings.hf_access_token or None,
            device=device,
            use_cache=(device.type == "cuda"),
        )
        self._device = device
        tensor = image_to_tensor(image, device=device)
        with torch.inference_mode():
            prediction = self._model([tensor])[0]
        return RawDetections(
            boxes=prediction["boxes"].detach().cpu(),
            scores=prediction["scores"].detach().cpu(),
            labels=prediction["labels"].detach().cpu(),
        )

    def explain(self, image: PILImage.Image, label: int) -> str | None:
        """Grad-CAM heatmap for one detection label (base64 PNG)."""
        if self._model is None or self._device is None:
            return None
        if self._gradcam is None or self._gradcam.model is not self._model:
            self._gradcam = GradCAM(self._model)
        tensor = image_to_tensor(image, device=self._device).unsqueeze(0)
        with torch.enable_grad():
            return self._gradcam.compute(tensor, label)


class YoloAdapter(ModelAdapter):
    """YOLOv8: PIL in, ultralytics preprocessing, no explainability."""

    model_type = "yolov8"
    model_version = YOLO_MODEL_VERSION

    def __init__(self) -> None:
        """Build the adapter; class names fill after the first forward."""
        self._class_names: dict[int, str] = {}

    @property
    def class_names(self) -> dict[int, str]:
        """Class-id to name map from the loaded YOLO model."""
        return self._class_names

    def forward(self, image: PILImage.Image, device: torch.device) -> RawDetections:
        """Preprocess and run Faster R-CNN forward on ``device``."""
        settings = get_settings()
        model = load_yolo(
            checkpoint_path=settings.yolo_checkpoint,
            use_cache=(device.type == "cuda"),
        )
        model.to(device)
        self._class_names = {int(i): name for i, name in model.names.items()}
        results = model(image, conf=0.0, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return RawDetections(
                boxes=torch.empty(0, 4),
                scores=torch.empty(0),
                labels=torch.empty(0, dtype=torch.int64),
            )
        boxes = results[0].boxes
        return RawDetections(
            boxes=boxes.xyxy.cpu().clone(),
            scores=boxes.conf.cpu().clone(),
            labels=boxes.cls.int().cpu().clone(),
        )


DEFAULT_ADAPTERS: dict[str, ModelAdapter] = {
    "faster-rcnn": FasterRCNNAdapter(),
    "yolov8": YoloAdapter(),
}
