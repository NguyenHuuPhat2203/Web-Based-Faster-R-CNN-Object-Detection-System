"""Faster R-CNN backbone factory and weight loading (no FastAPI / GradCAM)."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ml.core.constants import FRCNN_NUM_CLASSES

logger = structlog.get_logger(__name__)

_model_cache: torch.nn.Module | None = None


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_faster_rcnn(
    num_classes: int = FRCNN_NUM_CLASSES,
    *,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create a Faster R-CNN ResNet-50 FPN with a custom box predictor head.

    Args:
        num_classes: Number of classes including background.
        pretrained: When True, load COCO pretrained backbone weights.
            Set False for fast CPU unit/overfit tests.

    Returns:
        Model with replaced predictor head.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
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
        Model with loaded weights (strict=False).
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
    model.load_state_dict(state, strict=False)
    logger.info("loaded_frcnn_weights", path=str(checkpoint_path))
    return model


def load_faster_rcnn(
    *,
    checkpoint_path: Path | None = None,
    safetensors_path: Path | None = None,
    hf_repo: str | None = None,
    hf_filename: str | None = None,
    hf_token: str | None = None,
    num_classes: int = FRCNN_NUM_CLASSES,
    device: torch.device | None = None,
    use_cache: bool = True,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create and load a Faster R-CNN, preferring safetensors then local pth then HF.

    Args:
        checkpoint_path: Legacy ``.pth`` path.
        safetensors_path: Preferred ``.safetensors`` path (registry/local).
        hf_repo: Hugging Face repo id for fallback download.
        hf_filename: Filename inside the HF repo.
        hf_token: Optional HF token.
        num_classes: Classes including background.
        device: Target device.
        use_cache: Cache the loaded module process-wide.
        pretrained: Load COCO weights when no checkpoint is found.

    Returns:
        Eval-mode model on ``device``.
    """
    global _model_cache
    if use_cache and _model_cache is not None:
        return _model_cache

    target = device or get_device()
    model = create_faster_rcnn(num_classes=num_classes, pretrained=pretrained)

    loaded = False
    for path in (safetensors_path, checkpoint_path):
        if path is not None and path.exists():
            load_state_dict(model, path, device=target)
            loaded = True
            break

    if not loaded and hf_repo and hf_filename:
        try:
            from huggingface_hub import hf_hub_download

            logger.info("downloading_frcnn_from_hf", repo=hf_repo, filename=hf_filename)
            hf_path = Path(
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_filename,
                    token=hf_token or None,
                )
            )
            load_state_dict(model, hf_path, device=target)
            loaded = True
        except Exception as exc:
            logger.warning("frcnn_checkpoint_download_failed", error=str(exc))

    model.to(target)
    model.eval()
    if use_cache:
        _model_cache = model
    return model


def reset_model_cache() -> None:
    """Clear the cached Faster R-CNN module (tests)."""
    global _model_cache
    _model_cache = None
