"""Shared domain constants for brain-tumor detection."""

from __future__ import annotations

# Dataset (ultralytics brain-tumor.yaml) classes: 0=negative, 1=positive.
# Faster R-CNN reserves 0 for background, so dataset class k becomes k + 1.
FASTER_RCNN_NUM_CLASSES = 3
FASTER_RCNN_CLASS_NAMES: dict[int, str] = {1: "negative", 2: "positive"}
FASTER_RCNN_MODEL_VERSION = "resnet50-fpn-v1"
YOLO_MODEL_VERSION = "yolov8n"
DEFAULT_SCORE_THRESHOLD = 0.5
