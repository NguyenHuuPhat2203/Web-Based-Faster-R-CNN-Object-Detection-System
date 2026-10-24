"""Tests for health and metrics endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def test_healthz_includes_model_version(client: TestClient) -> None:
    """Health probe exposes model_version."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "gpu_available" in data


def test_readyz_ok(client: TestClient) -> None:
    """Readiness probe succeeds when the database answers."""
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_unavailable_when_db_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness returns 503 when the database cannot be queried."""

    def _broken_session_local():
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        session.execute.side_effect = RuntimeError("db down")
        return session

    monkeypatch.setattr("backend.main.SessionLocal", _broken_session_local)

    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_health_removed(client: TestClient) -> None:
    """Legacy ``/health`` alias is no longer registered."""
    response = client.get("/health")
    assert response.status_code == 404


def test_metrics_prometheus(client: TestClient) -> None:
    """``/metrics`` returns Prometheus text format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "inference_latency_ms" in body


def test_correlation_id_echoed_or_replaced(client: TestClient) -> None:
    """A well-formed correlation id is echoed; a malformed one is replaced."""
    ok = client.get("/healthz", headers={"x-correlation-id": "abc-123"})
    assert ok.headers["x-correlation-id"] == "abc-123"
    bad = client.get("/healthz", headers={"x-correlation-id": "bad id\twith space"})
    assert bad.headers["x-correlation-id"] != "bad id\twith space"
    assert len(bad.headers["x-correlation-id"]) == 36


def test_unknown_predict_path_adds_no_metric_label(client: TestClient) -> None:
    """Arbitrary /predict/* paths must not create new metric label values."""
    client.get("/predict/some-random-path-123")
    assert "some-random-path-123" not in client.get("/metrics").text
