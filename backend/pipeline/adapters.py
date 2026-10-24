"""Model adapters for Faster R-CNN and YOLO."""

from __future__ import annotations

import torch
from PIL import Image as PILImage

from ml.core.constants import (
    FASTER_RCNN_CLASS_NAMES,
    FASTER_RCNN_MODEL_VERSION,
    YOLO_MODEL_VERSION,
)
from ml.models.faster_rcnn import get_device, load_faster_rcnn
from ml.models.yolo import load_yolo
from ml.transforms import image_to_tensor

from ..config import get_settings
from .gradcam import GradCAM
from .inference import ModelAdapter, RawDetections


def _is_primary(device: torch.device) -> bool:
    """Only models on the primary device are cached; fallbacks load fresh."""
    return device == get_device()


class FasterRCNNAdapter(ModelAdapter):
    """Faster R-CNN: tensor preprocessing + optional Grad-CAM."""

    model_type = "faster-rcnn"
    model_version = FASTER_RCNN_MODEL_VERSION
    class_names = FASTER_RCNN_CLASS_NAMES

    def __init__(self) -> None:
        """Build the adapter; the model loads lazily on first forward."""
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def forward(self, image: PILImage.Image, device: torch.device) -> RawDetections:
        """Preprocess and run Faster R-CNN forward on ``device``."""
        settings = get_settings()
        self._model = load_faster_rcnn(
            checkpoint_path=settings.faster_rcnn_checkpoint,
            safetensors_path=settings.faster_rcnn_safetensors,
            device=device,
            use_cache=_is_primary(device),
        )
        self._device = device
        tensor = image_to_tensor(image, device=device)
        with torch.inference_mode():
            prediction = self._model([tensor])[0]
        return RawDetections(
            boxes=prediction["boxes"].cpu(),
            scores=prediction["scores"].cpu(),
            labels=prediction["labels"].cpu(),
        )

    def explain(self, image: PILImage.Image, label: int) -> str | None:
        """Grad-CAM heatmap for the best detection of ``label`` (base64 PNG)."""
        if self._model is None or self._device is None:
            return None
        tensor = image_to_tensor(image, device=self._device)
        return GradCAM(self._model).compute(tensor, label)


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
        """Run YOLOv8 (ultralytics preprocessing) on ``device``."""
        model = load_yolo(
            checkpoint_path=get_settings().yolo_checkpoint,
            use_cache=_is_primary(device),
        )
        self._class_names = {int(i): name for i, name in model.names.items()}
        ultralytics_device = (device.index or 0) if device.type == "cuda" else "cpu"
        results = model(image, conf=0.0, device=ultralytics_device, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return RawDetections(
                boxes=torch.empty(0, 4),
                scores=torch.empty(0),
                labels=torch.empty(0, dtype=torch.int64),
            )
        return RawDetections(
            boxes=boxes.xyxy.cpu().clone(),
            scores=boxes.conf.cpu().clone(),
            labels=boxes.cls.long().cpu(),
        )


DEFAULT_ADAPTERS: dict[str, ModelAdapter] = {
    "faster-rcnn": FasterRCNNAdapter(),
    "yolov8": YoloAdapter(),
}
