"""YOLO -> Faster R-CNN target conversion and trainer accumulation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.config import TrainConfig
from training.data.dataset import YoloDetectionDataset
from training.trainer import trainer as trainer_module


def _make_dataset(tmp_path: Path, label_text: str) -> YoloDetectionDataset:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 50)).save(images / "a.png")
    (labels / "a.txt").write_text(label_text, encoding="utf-8")
    return YoloDetectionDataset(images, labels)


def test_yolo_boxes_become_absolute_xyxy_with_background_offset(tmp_path) -> None:
    """Normalized cx,cy,w,h scale by (W,H); YOLO class 1 becomes label 2."""
    dataset = _make_dataset(tmp_path, "1 0.5 0.5 0.2 0.4\n")
    _, target = dataset[0]
    assert target["boxes"][0].tolist() == pytest.approx([40.0, 15.0, 60.0, 35.0])
    assert target["labels"].tolist() == [2]


def test_boxes_are_clipped_and_degenerate_boxes_dropped(tmp_path) -> None:
    """Out-of-frame boxes are clipped; boxes with no area are removed."""
    dataset = _make_dataset(tmp_path, "0 0.95 0.5 0.2 0.4\n0 0.5 0.5 0.0 0.4\n")
    _, target = dataset[0]
    assert target["boxes"][0].tolist() == pytest.approx([85.0, 15.0, 100.0, 35.0])
    assert target["labels"].tolist() == [1]


def test_unknown_class_id_fails_loudly(tmp_path) -> None:
    """A class id beyond the model head must not reach the loss as a CUDA assert."""
    dataset = _make_dataset(tmp_path, "2 0.5 0.5 0.2 0.4\n")
    with pytest.raises(ValueError, match="class id"):
        dataset[0]


class _ScalarModel(torch.nn.Module):
    """Loss equals one scalar parameter, so each batch contributes gradient 1."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, images, targets):
        return {"loss": self.weight * 1.0}


def test_partial_accumulation_window_still_steps(tmp_path, monkeypatch) -> None:
    """3 batches with accumulation 2: the trailing batch must update the weights."""
    monkeypatch.setattr(
        trainer_module, "create_faster_rcnn", lambda **_: _ScalarModel()
    )
    batches = [([torch.zeros(3, 4, 4)], [{}])] * 3
    config = TrainConfig(
        learning_rate=1.0,
        momentum=0.0,
        weight_decay=0.0,
        epochs=1,
        gradient_accumulation_steps=2,
    )
    best = trainer_module.train_faster_rcnn(
        batches, batches, config, output_dir=tmp_path
    )
    # window 1: (1/2 + 1/2) -> -1.0; trailing window: 1/2 -> -0.5
    assert torch.load(best, weights_only=True)["weight"].item() == pytest.approx(8.5)
