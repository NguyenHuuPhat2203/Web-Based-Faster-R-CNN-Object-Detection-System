"""Fixtures for HTTP integration tests (under backend/tests/integration)."""

from __future__ import annotations

import pytest

from ml.core.entities import InferenceResult


@pytest.fixture(autouse=True)
def mock_ml_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading real torch checkpoints during HTTP integration tests."""

    def _mock_engine_run(
        self,
        contents: bytes,
        *,
        score_floor: float = 0.0,
        explain: bool = False,
    ) -> InferenceResult:
        return InferenceResult(
            model_type=self.adapter.model_type,
            model_version="test",
            boxes=[[10, 20, 100, 120], [30, 40, 90, 110]],
            labels=[1, 1],
            scores=[0.9, 0.3],
            label_names=["Brain tumor", "Brain tumor"],
        )

    monkeypatch.setattr(
        "backend.services.prediction_service.InferenceEngine.run", _mock_engine_run
    )
