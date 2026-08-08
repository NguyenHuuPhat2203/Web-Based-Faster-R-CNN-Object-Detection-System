"""Pure post-processing helpers (no nn.Module, no gradients)."""

from __future__ import annotations

import torch
from torchvision.ops import nms

from ml.core.constants import FRCNN_CLASS_NAMES
from ml.core.entities import InferenceResult


def filter_by_threshold(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep detections with score above threshold.

    Args:
        boxes: Boxes tensor ``[N, 4]`` in xyxy format.
        scores: Confidence scores ``[N]``.
        labels: Class indices ``[N]``.
        threshold: Minimum confidence.

    Returns:
        Filtered ``(boxes, scores, labels)`` tensors.
    """
    keep = scores > threshold
    return boxes[keep], scores[keep], labels[keep]


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-agnostic NMS and return surviving detections."""
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], labels[keep]


def tensors_to_inference_result(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_names: dict[int, str] | None = None,
    model_type: str,
    model_version: str,
    heatmap: str | None = None,
) -> InferenceResult:
    """Convert filtered tensors into an ``InferenceResult``."""
    names = class_names or FRCNN_CLASS_NAMES
    box_list = boxes.detach().cpu().tolist() if boxes.numel() else []
    label_list = (
        [int(x) for x in labels.detach().cpu().tolist()] if labels.numel() else []
    )
    score_list = (
        [float(x) for x in scores.detach().cpu().tolist()] if scores.numel() else []
    )
    label_names = [names.get(label, f"Class {label}") for label in label_list]
    return InferenceResult(
        boxes=box_list,
        labels=label_list,
        scores=score_list,
        label_names=label_names,
        heatmap=heatmap,
        model_type=model_type,
        model_version=model_version,
    )
