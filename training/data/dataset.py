"""YOLO-format detection dataset for Faster R-CNN training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.transforms import image_to_tensor


class YoloDetectionDataset(Dataset):
    """Load images + YOLO txt labels and return torchvision detection targets."""

    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        """Initialize dataset from YOLO folder layout.

        Args:
            images_dir: Directory of image files.
            labels_dir: Directory of matching ``.txt`` label files.
        """
        self.images = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        self.labels_dir = labels_dir

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(image_tensor, target)`` for Faster R-CNN."""
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = image_to_tensor(image)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                # Faster R-CNN: 0 = background; shift YOLO class by +1
                labels.append(int(cls) + 1)

        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([index]),
        }
        return tensor, target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate variable-size detection samples."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
