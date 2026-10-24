"""YOLOv8 thin wrapper (Ultralytics) — no FastAPI dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_model: Any = None


def _verify_yolo_weights_digest(checkpoint_path: Path) -> None:
    """Fail closed when sibling metadata.json records a SHA-256 digest."""
    meta_path = checkpoint_path.parent / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("YOLO weights digest mismatch") from exc
    if not isinstance(meta, dict):
        raise RuntimeError("YOLO weights digest mismatch")
    if "weights_sha256" not in meta:
        return
    expected = meta["weights_sha256"]
    if not isinstance(expected, str):
        raise RuntimeError("YOLO weights digest mismatch")
    with checkpoint_path.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    if digest != expected:
        raise RuntimeError("YOLO weights digest mismatch")


def load_yolo(*, checkpoint_path: Path, use_cache: bool = True) -> Any:
    """Load the tumor-trained Ultralytics YOLO model.

    There is deliberately no fallback to stock ``yolov8n.pt``: COCO classes
    would be reported as tumor findings.

    Args:
        checkpoint_path: Local ``.pt`` checkpoint.
        use_cache: Return / store the process-wide cached model. Callers must
            only cache models on the primary device.

    Returns:
        Loaded YOLO model instance.

    Raises:
        RuntimeError: If ultralytics is not installed or the weights digest mismatches.
        FileNotFoundError: If the checkpoint is missing.
    """
    global _model
    if use_cache and _model is not None:
        return _model

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLOv8 requires the 'ultralytics' package. "
            "Install it with: pip install ultralytics"
        ) from exc

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint_path}")
    _verify_yolo_weights_digest(checkpoint_path)
    model = YOLO(str(checkpoint_path))
    logger.info("loaded_yolo_weights", path=str(checkpoint_path))
    if use_cache:
        _model = model
    return model


def reset_model_cache() -> None:
    """Clear the cached YOLO model (tests)."""
    global _model
    _model = None
