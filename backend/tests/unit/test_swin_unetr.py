"""SwinUNETR loading, intensity normalization, and channel decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.models.swin_unetr import SwinUNETRSegmentor, decode_channels, normalize_nonzero


def test_decode_channels_nested_regions() -> None:
    """WT-only -> ED(2), TC-only -> NCR(1), ET -> ET(4), outside -> 0."""
    binary = np.zeros((3, 4, 1, 1), dtype=bool)
    binary[1, 1:] = True  # WT covers voxels 1..3
    binary[0, 2:] = True  # TC covers voxels 2..3
    binary[2, 3:] = True  # ET covers voxel 3
    assert decode_channels(binary).ravel().tolist() == [0, 2, 1, 4]


def test_normalize_nonzero_keeps_background_zero() -> None:
    """Only foreground voxels are z-scored; background stays exactly 0."""
    volume = np.array([[0.0, 2.0, 4.0, 0.0]], dtype=np.float32)
    normalize_nonzero(volume)
    assert volume.tolist() == [[0.0, -1.0, 1.0, 0.0]]


def test_build_model_uses_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch.load must be called with weights_only=True (no real checkpoint)."""
    captured: dict[str, object] = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return {"state_dict": {}}

    class _Dummy:
        def load_state_dict(self, state):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr("ml.models.swin_unetr.torch.load", fake_load)
    monkeypatch.setattr(
        "ml.models.swin_unetr.SwinUNETR", lambda *args, **kwargs: _Dummy()
    )

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    SwinUNETRSegmentor(weights_path=weights)
    assert captured.get("weights_only") is True


def _patch_swin_load(monkeypatch: pytest.MonkeyPatch, payload):
    class _Dummy:
        def load_state_dict(self, state):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr("ml.models.swin_unetr.torch.load", lambda *a, **k: payload)
    monkeypatch.setattr(
        "ml.models.swin_unetr.SwinUNETR", lambda *args, **kwargs: _Dummy()
    )


def test_build_model_rejects_missing_state_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint without state_dict must fail closed."""
    _patch_swin_load(monkeypatch, {"not_state_dict": {}})
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="Invalid SwinUNETR checkpoint"):
        SwinUNETRSegmentor(weights_path=weights)
