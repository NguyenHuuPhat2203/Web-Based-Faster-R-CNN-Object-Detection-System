"""Shared domain constants for brain-tumor detection."""

from __future__ import annotations

FRCNN_NUM_CLASSES = 3
FRCNN_CLASS_NAMES: dict[int, str] = {1: "Brain tumor", 2: "Brain tumor"}
FRCNN_MODEL_VERSION = "resnet50-fpn-v1"
YOLO_MODEL_VERSION = "yolov8n"
DEFAULT_SCORE_THRESHOLD = 0.5
