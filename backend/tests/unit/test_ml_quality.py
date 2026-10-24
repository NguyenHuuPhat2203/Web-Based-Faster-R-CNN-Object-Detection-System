"""ML quality tests: overfit, edge cases, golden values, performance budgets."""

from __future__ import annotations

import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ml.models.faster_rcnn import create_faster_rcnn, reset_model_cache


class _TinyDetectionDataset(Dataset):
    """Synthetic single-sample detection dataset for overfit tests."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torch.rand(3, 64, 64)
        boxes = torch.tensor([[8.0, 8.0, 40.0, 40.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        return image, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }


def _collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


@pytest.mark.overfit
def test_overfit_single_batch() -> None:
    """MobileNet Faster R-CNN must overfit one tiny batch on CPU."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    reset_model_cache()
    device = torch.device("cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    model = model.to(device)

    loader = DataLoader(
        _TinyDetectionDataset(size=2),
        batch_size=2,
        collate_fn=_collate,
    )
    images, targets = next(iter(loader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01,
        momentum=0.9,
    )
    model.train()
    initial_loss: float | None = None
    best_loss = float("inf")
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss_value = float(loss.item())
        if not torch.isfinite(loss).item():
            break
        if initial_loss is None:
            initial_loss = loss_value
        best_loss = min(best_loss, loss_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if best_loss < 0.5:
            break

    assert initial_loss is not None, "No finite loss observed"
    assert best_loss < initial_loss * 0.75 or best_loss < 1.0, (
        f"Overfit test FAILED: initial={initial_loss:.4f} best={best_loss:.4f}"
    )


@pytest.mark.edge
def test_single_element_batch_inference() -> None:
    """Batch size 1 must not crash under inference_mode."""
    reset_model_cache()
    device = torch.device("cpu")
    model = create_faster_rcnn(num_classes=3, pretrained=False).to(device)
    model.eval()
    image = torch.rand(3, 64, 64, device=device)
    with torch.inference_mode():
        outputs = model([image])
    assert isinstance(outputs, list)
    assert "boxes" in outputs[0]


@pytest.mark.perf
def test_inference_latency_budget() -> None:
    """Tiny forward pass stays within a generous CPU latency budget."""
    reset_model_cache()
    device = torch.device("cpu")
    model = create_faster_rcnn(num_classes=3, pretrained=False).to(device)
    model.eval()
    image = torch.rand(3, 64, 64, device=device)

    with torch.inference_mode():
        for _ in range(2):
            _ = model([image])

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            _ = model([image])
    avg_ms = (time.perf_counter() - started) / 3 * 1000
    assert avg_ms <= 60_000, f"Latency budget exceeded: {avg_ms:.1f}ms"


@pytest.mark.golden
def test_create_model_forward_golden_shape() -> None:
    """Untrained model on fixed seed input yields expected output keys/shapes."""
    reset_model_cache()
    torch.manual_seed(42)
    model = create_faster_rcnn(num_classes=3, pretrained=False)
    model.eval()
    image = torch.rand(3, 64, 64)
    with torch.inference_mode():
        outputs = model([image])
    assert len(outputs) == 1
    assert outputs[0]["boxes"].ndim == 2
    assert outputs[0]["boxes"].shape[-1] == 4
    assert outputs[0]["scores"].ndim == 1
    assert outputs[0]["labels"].ndim == 1
