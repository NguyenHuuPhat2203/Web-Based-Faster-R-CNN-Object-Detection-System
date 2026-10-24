"""Pure post-processing helpers (no nn.Module, no gradients)."""

from __future__ import annotations

import torch

from ml.core.entities import InferenceResult


def filter_above_floor(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep detections with score at or above the floor.

    Args:
        boxes: Boxes tensor ``[N, 4]`` in xyxy format.
        scores: Confidence scores ``[N]``.
        labels: Class indices ``[N]``.
        floor: Minimum confidence to return.

    Returns:
        Filtered ``(boxes, scores, labels)`` tensors.
    """
    keep = scores >= floor
    return boxes[keep], scores[keep], labels[keep]


def tensors_to_inference_result(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_names: dict[int, str],
    model_type: str,
    model_version: str,
    heatmap: str | None = None,
) -> InferenceResult:
    """Convert filtered tensors into an ``InferenceResult``."""
    label_list = [int(label) for label in labels.tolist()]
    return InferenceResult(
        boxes=[[float(v) for v in box] for box in boxes.tolist()],
        labels=label_list,
        scores=[float(score) for score in scores.tolist()],
        label_names=[class_names.get(label, f"class_{label}") for label in label_list],
        heatmap=heatmap,
        model_type=model_type,
        model_version=model_version,
    )
