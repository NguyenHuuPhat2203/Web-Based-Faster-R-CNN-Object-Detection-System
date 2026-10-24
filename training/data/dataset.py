"""YOLO-format detection dataset for Faster R-CNN training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.core.constants import FASTER_RCNN_NUM_CLASSES
from ml.transforms import image_to_tensor
from training.data.layout import list_images


def _read_yolo_boxes(
    label_path: Path, width: int, height: int, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a YOLO txt file into absolute ``xyxy`` boxes and Faster R-CNN labels.

    YOLO class ``k`` becomes label ``k + 1`` because torchvision reserves 0 for
    background. Boxes are clipped to the image; boxes left with no area are dropped
    (torchvision raises on them).

    Raises:
        ValueError: On a malformed line or a class id outside the model's classes.
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}: expected 5 fields, got {line!r}")
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(float, parts[1:])
            label = class_id + 1
            if not 1 <= label < num_classes:
                raise ValueError(
                    f"{label_path}: class id {class_id} outside 0..{num_classes - 2}"
                )
            x_min = max((center_x - box_width / 2) * width, 0.0)
            y_min = max((center_y - box_height / 2) * height, 0.0)
            x_max = min((center_x + box_width / 2) * width, float(width))
            y_max = min((center_y + box_height / 2) * height, float(height))
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.int64
        )
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
    )


class YoloDetectionDataset(Dataset):
    """Load images + YOLO txt labels and return torchvision detection targets."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        num_classes: int = FASTER_RCNN_NUM_CLASSES,
    ) -> None:
        """Initialize dataset from YOLO folder layout.

        Args:
            images_dir: Directory of image files.
            labels_dir: Directory of matching ``.txt`` label files.
            num_classes: Model classes including background.
        """
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir
        self.num_classes = num_classes

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(image_tensor, target)`` for Faster R-CNN."""
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = _read_yolo_boxes(
            self.labels_dir / f"{image_path.stem}.txt",
            width,
            height,
            self.num_classes,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }
        return image_to_tensor(image), target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate variable-size detection samples."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
