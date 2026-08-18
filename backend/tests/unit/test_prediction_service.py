"""Unit tests for the prediction seam: injected adapter + Scan persistence."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from backend.models import Image, User
from backend.services.auth_service import AuthService
from backend.services.prediction_service import PredictionService
from ml.core.entities import Scan


class _FakeAdapter:
    """Minimal ModelAdapter stand-in: fixed detection, no real model."""

    model_type = "fake"
    model_version = "0.0.1"
    class_names: dict[int, str] = {1: "tumor"}

    def forward(self, image: PILImage.Image, device) -> object:
        import torch

        return type(
            "Raw",
            (),
            {
                "boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]]),
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([1]),
            },
        )()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_user(db) -> User:
    return AuthService(db).register(
        email="seam@test.dev", username="seam", password="password123"
    )


def test_scan_entity_carries_the_artifact() -> None:
    """Scan bundles the upload, model meta, and detections."""
    scan = Scan(
        contents=b"img",
        original_name="mri.png",
        content_type="image/png",
        threshold=0.5,
        model_type="faster-rcnn",
        model_version="v1",
        detections=[],
    )
    assert scan.threshold == 0.5
    assert scan.original_name == "mri.png"
    assert scan.content_type == "image/png"


def test_run_persists_scan_with_injected_adapter(db_session) -> None:
    """A fake adapter flows through run() and the scan is persisted."""
    user = _make_user(db_session)
    service = PredictionService(db_session, {"fake": _FakeAdapter()})

    result = service.run(
        model_type="fake",
        user=user,
        contents=_png_bytes(),
        original_name="mri.png",
        content_type="image/png",
        score_floor=0.1,
    )

    assert result.boxes == [[10.0, 20.0, 60.0, 80.0]]
    assert result.scores == pytest.approx([0.95])
    saved = db_session.query(Image).filter(Image.user_id == user.id).one()
    assert saved.original_name == "mri.png"
    assert saved.model_type == "fake"
    assert saved.threshold == 0.1
    assert len(saved.detections) == 1


def test_run_requires_adapter_for_unknown_model(db_session) -> None:
    """Unknown model_type raises against the injected adapter map."""
    user = _make_user(db_session)
    service = PredictionService(db_session, {"fake": _FakeAdapter()})

    with pytest.raises(KeyError):
        service.run(
            model_type="nope",
            user=user,
            contents=_png_bytes(),
            original_name="mri.png",
            content_type="image/png",
            score_floor=0.1,
        )
