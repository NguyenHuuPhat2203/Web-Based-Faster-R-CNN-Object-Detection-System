"""Unit tests for OOM -> CPU fallback in the inference engine."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image as PILImage

from backend.pipeline.inference import InferenceEngine, ModelAdapter, RawDetections


def _png_bytes(size: int = 64) -> bytes:
    """Return a minimal grey PNG."""
    img = PILImage.new("RGB", (size, size), (80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _OOMAdapter(ModelAdapter):
    """Raise OOM twice, then return one detection."""

    model_type = "faster-rcnn"
    model_version = "test"
    class_names = {1: "Brain tumor"}

    def __init__(self) -> None:
        self.calls = 0
        self.devices: list[str] = []

    def forward(self, image, device) -> RawDetections:
        self.calls += 1
        self.devices.append(str(device))
        if self.calls <= 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return RawDetections(
            boxes=torch.tensor([[10.0, 20.0, 100.0, 120.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
        )


def test_engine_falls_back_to_cpu_after_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two GPU OOMs must trigger a fresh CPU forward instead of raising."""
    monkeypatch.setattr(
        "backend.pipeline.inference.get_device", lambda: torch.device("cuda")
    )

    adapter = _OOMAdapter()
    result = InferenceEngine(adapter).run(_png_bytes(), score_floor=0.5, explain=True)

    assert adapter.calls == 3  # 2 GPU attempts + 1 CPU fallback
    assert adapter.devices[-1] == "cpu"
    assert len(result.detections) == 1
    assert result.heatmap is None  # explain skipped after CPU fallback


def test_engine_reraises_non_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-OOM failures keep raising."""

    class _ExplodingAdapter(ModelAdapter):
        def forward(self, image, device) -> RawDetections:
            raise RuntimeError("model exploded")

    monkeypatch.setattr(
        "backend.pipeline.inference.get_device", lambda: torch.device("cuda")
    )

    with pytest.raises(RuntimeError, match="model exploded"):
        InferenceEngine(_ExplodingAdapter()).run(_png_bytes())


def test_frcnn_adapter_loads_fresh_cpu_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 invariant: CPU fallback loads an uncached model (use_cache=False)."""
    import backend.pipeline.adapters as adapters

    calls: list[tuple] = []

    class _DummyModel:
        def __call__(self, tensor):
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
            ]

    def fake_load(**kwargs):
        calls.append((kwargs.get("device"), kwargs.get("use_cache")))
        return _DummyModel()

    monkeypatch.setattr(adapters, "load_faster_rcnn", fake_load)

    adapter = adapters.FasterRCNNAdapter()
    adapter.forward(PILImage.new("RGB", (8, 8)), torch.device("cpu"))

    assert calls[-1] == (torch.device("cpu"), False)
