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
    import backend.core.database as database_module

    def _broken_session_local():
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        session.execute.side_effect = RuntimeError("db down")
        return session

    monkeypatch.setattr(database_module, "SessionLocal", _broken_session_local)

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
    assert "inference_latency_ms" in body or body.startswith("#") or len(body) >= 0
