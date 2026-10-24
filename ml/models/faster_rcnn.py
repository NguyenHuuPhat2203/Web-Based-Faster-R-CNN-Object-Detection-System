"""Faster R-CNN backbone factory and weight loading (no FastAPI / GradCAM)."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ml.core.constants import FASTER_RCNN_NUM_CLASSES

logger = structlog.get_logger(__name__)

_model_cache: torch.nn.Module | None = None


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_faster_rcnn(
    num_classes: int = FASTER_RCNN_NUM_CLASSES,
    *,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create a Faster R-CNN ResNet-50 FPN with a custom box predictor head.

    Args:
        num_classes: Number of classes including background.
        pretrained: When True, start from COCO detection weights (training).
            When False, build an uninitialised network without downloading
            any weights (serving, tests): the checkpoint supplies every tensor.

    Returns:
        Model with replaced predictor head.
    """
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_state_dict(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load weights from ``.safetensors`` or ``.pth`` into ``model``.

    Args:
        model: Target module.
        checkpoint_path: Path to weight file.
        device: Map location for tensors.

    Returns:
        Model with loaded weights.

    Raises:
        RuntimeError: If checkpoint keys or shapes do not match the model.
    """
    target = device or get_device()
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path), device=str(target))
    else:
        state = torch.load(checkpoint_path, map_location=target, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    model.load_state_dict(state, strict=True)
    logger.info("loaded_faster_rcnn_weights", path=str(checkpoint_path))
    return model


def load_faster_rcnn(
    *,
    checkpoint_path: Path | None = None,
    safetensors_path: Path | None = None,
    num_classes: int = FASTER_RCNN_NUM_CLASSES,
    device: torch.device | None = None,
    use_cache: bool = True,
) -> torch.nn.Module:
    """Create and load a Faster R-CNN from local safetensors or ``.pth``.

    Args:
        checkpoint_path: Legacy ``.pth`` path.
        safetensors_path: Preferred ``.safetensors`` path (registry/local).
        num_classes: Classes including background.
        device: Target device.
        use_cache: Return / store the process-wide cached module. Callers must
            only cache models on the primary device.

    Returns:
        Eval-mode model on ``device``.

    Raises:
        FileNotFoundError: If neither safetensors nor a local ``.pth`` exists.
    """
    global _model_cache
    if use_cache and _model_cache is not None:
        return _model_cache

    weights_path = next(
        (p for p in (safetensors_path, checkpoint_path) if p and p.is_file()), None
    )
    if weights_path is None:
        raise FileNotFoundError("Faster R-CNN weights not found")

    target = device or get_device()
    model = create_faster_rcnn(num_classes=num_classes, pretrained=False)
    load_state_dict(model, weights_path, device=target)

    model.to(target)
    model.eval()
    if use_cache:
        _model_cache = model
    return model


def reset_model_cache() -> None:
    """Clear the cached Faster R-CNN module (tests)."""
    global _model_cache
    _model_cache = None
