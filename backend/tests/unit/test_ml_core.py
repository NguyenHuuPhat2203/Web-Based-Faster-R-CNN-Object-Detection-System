"""Unit tests for shared ML core helpers."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from ml.core.postprocess import filter_above_floor, tensors_to_inference_result
from ml.transforms import image_to_tensor


def test_filter_above_floor() -> None:
    """Scores below the floor are dropped; at the floor are kept."""
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
    )
    scores = torch.tensor([0.9, 0.5, 0.1])
    labels = torch.tensor([1, 1, 1])
    out_boxes, out_scores, out_labels = filter_above_floor(
        boxes, scores, labels, floor=0.5
    )
    assert out_boxes.shape[0] == 2
    assert float(out_scores[0]) == pytest.approx(0.9, rel=1e-5)
    assert float(out_scores[1]) == pytest.approx(0.5, rel=1e-5)  # == floor kept
    assert int(out_labels[0]) == 1


def test_tensors_to_inference_result() -> None:
    """Tensor conversion produces matching list lengths."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.8])
    labels = torch.tensor([1])
    result = tensors_to_inference_result(
        boxes,
        scores,
        labels,
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
    )
    assert len(result.boxes) == 1
    assert result.label_names[0] == "Brain tumor"
    assert len(result.detections) == 1


def test_image_to_tensor_rgb() -> None:
    """Inference transform yields CHW RGB float tensor."""
    image = Image.new("RGB", (32, 48), color=(128, 128, 128))
    tensor = image_to_tensor(image)
    assert tuple(tensor.shape) == (3, 48, 32)
    assert tensor.dtype == torch.float32
