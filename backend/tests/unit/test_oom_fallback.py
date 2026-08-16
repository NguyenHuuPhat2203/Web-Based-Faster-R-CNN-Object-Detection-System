"""Unit tests for OOM -> CPU fallback in both inference pipelines."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image as PILImage

from backend.pipeline.faster_rcnn_pipeline import run_faster_rcnn
from backend.pipeline.yolo_pipeline import run_yolo


def _png_bytes(size: int = 64) -> bytes:
    """Return a minimal grey PNG."""
    img = PILImage.new("RGB", (size, size), (80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_frcnn_falls_back_to_cpu_after_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two GPU OOMs must trigger a fresh CPU forward instead of raising."""
    import backend.pipeline.faster_rcnn_pipeline as mod

    calls = {"forwards": 0, "cpu_load": False}

    class Dummy:
        pass

    dummy = Dummy()

    def fake_load(**kwargs):
        if kwargs.get("device") == "cpu":
            calls["cpu_load"] = True
        return dummy

    def fake_load_model_and_gradcam():
        return dummy, None

    def fake_forward(model, image_tensor):
        calls["forwards"] += 1
        if calls["forwards"] <= 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 100.0, 120.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]

    monkeypatch.setattr(mod, "load_faster_rcnn", fake_load)
    monkeypatch.setattr(mod, "_load_model_and_gradcam", fake_load_model_and_gradcam)
    monkeypatch.setattr(mod, "_forward", fake_forward)

    pil_image, result = run_faster_rcnn(_png_bytes(), threshold=0.5, explain=True)

    assert calls["forwards"] == 3  # 2 GPU attempts + 1 CPU fallback
    assert calls["cpu_load"] is True
    assert len(result.detections) == 1
    assert result.heatmap is None  # Grad-CAM skipped in fallback


def test_frcnn_still_raises_on_non_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-OOM failures keep raising (v2 INFERENCE_FAILED path)."""
    import backend.pipeline.faster_rcnn_pipeline as mod

    class Dummy:
        pass

    def fake_load_model_and_gradcam():
        return Dummy(), None

    def fake_forward(model, image_tensor):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(mod, "_load_model_and_gradcam", fake_load_model_and_gradcam)
    monkeypatch.setattr(mod, "_forward", fake_forward)

    with pytest.raises(RuntimeError, match="model exploded"):
        run_faster_rcnn(_png_bytes())


def test_yolo_falls_back_to_cpu_after_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two OOMs must trigger a fresh CPU YOLO forward instead of raising."""
    import backend.pipeline.yolo_pipeline as mod

    calls = {"forwards": 0}

    class DummyBoxes:
        xyxy = torch.tensor([[10.0, 20.0, 100.0, 120.0]])
        conf = torch.tensor([0.9])
        cls = torch.tensor([0])

        def __len__(self):
            return 1

    class DummyResult:
        boxes = DummyBoxes()

    class DummyModel:
        names = {0: "Brain tumor"}

        def to(self, device):
            return self

        def __call__(self, *args, **kwargs):
            calls["forwards"] += 1
            if calls["forwards"] <= 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return [DummyResult()]

    dummy = DummyModel()

    monkeypatch.setattr(mod, "load_yolo", lambda **kwargs: dummy)
    monkeypatch.setattr(mod, "get_class_names", lambda: {0: "Brain tumor"})

    pil_image, result = run_yolo(_png_bytes(), threshold=0.5)

    assert calls["forwards"] == 3  # 2 GPU attempts + 1 CPU fallback
    assert len(result.detections) == 1
