"""YOLOv8 thin wrapper (Ultralytics) — no FastAPI dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_model: Any = None
_class_names: dict[int, str] = {}


def load_yolo(
    *,
    checkpoint_path: Path | None = None,
    fallback_weights: str = "yolov8n.pt",
    use_cache: bool = True,
) -> Any:
    """Load an Ultralytics YOLO model.

    Args:
        checkpoint_path: Local ``.pt`` checkpoint.
        fallback_weights: Ultralytics weight name when local file is missing.
        use_cache: Cache the loaded model process-wide.

    Returns:
        Loaded YOLO model instance.

    Raises:
        RuntimeError: If ultralytics is not installed.
    """
    global _model, _class_names
    if use_cache and _model is not None:
        return _model

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLOv8 requires the 'ultralytics' package. "
            "Install it with: pip install ultralytics"
        ) from exc

    if checkpoint_path is not None and checkpoint_path.exists():
        model = YOLO(str(checkpoint_path))
        logger.info("loaded_yolo_weights", path=str(checkpoint_path))
    else:
        model = YOLO(fallback_weights)
        logger.info("loaded_yolo_fallback", weights=fallback_weights)

    _class_names = {index: name for index, name in model.names.items()}
    if use_cache:
        _model = model
    return model


def get_class_names() -> dict[int, str]:
    """Return the class-name map for the loaded YOLO model."""
    if not _class_names:
        load_yolo()
    return _class_names


def reset_model_cache() -> None:
    """Clear the cached YOLO model (tests)."""
    global _model, _class_names
    _model = None
    _class_names = {}
