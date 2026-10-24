"""Unit tests for shared ML core helpers."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from ml.core.constants import FASTER_RCNN_CLASS_NAMES
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


def test_tensors_to_inference_result_maps_dataset_classes() -> None:
    """Faster R-CNN label 1 is the dataset's negative class, 2 the positive."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]])
    scores = torch.tensor([0.8, 0.6])
    labels = torch.tensor([2, 1])
    result = tensors_to_inference_result(
        boxes,
        scores,
        labels,
        class_names=FASTER_RCNN_CLASS_NAMES,
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
    )
    assert result.label_names == ["positive", "negative"]
    assert [d.label for d in result.detections] == [2, 1]


def test_image_to_tensor_rgb() -> None:
    """Inference transform yields CHW RGB float tensor."""
    image = Image.new("RGB", (32, 48), color=(128, 128, 128))
    tensor = image_to_tensor(image)
    assert tuple(tensor.shape) == (3, 48, 32)
    assert tensor.dtype == torch.float32
