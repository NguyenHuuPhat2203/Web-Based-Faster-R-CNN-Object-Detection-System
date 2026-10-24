"""Auth endpoint rate-limit behavior."""

from fastapi.testclient import TestClient

from backend.core.rate_limit import reset_auth_rate_limit


def test_login_rate_limited_on_third_hit(client: TestClient, monkeypatch) -> None:
    """Third /auth/login inside the window is 429 when MAX_HITS is 2."""
    monkeypatch.setattr("backend.core.rate_limit.MAX_HITS", 2)
    reset_auth_rate_limit()
    body = {"email": "rate@example.com", "password": "pass123456"}
    first = client.post("/auth/login", json=body)
    second = client.post("/auth/login", json=body)
    third = client.post("/auth/login", json=body)
    assert first.status_code != 429
    assert second.status_code != 429
    assert third.status_code == 429


def test_429_carries_retry_after(client: TestClient, monkeypatch) -> None:
    """Rejected requests tell the client when to retry."""
    monkeypatch.setattr("backend.core.rate_limit.MAX_HITS", 1)
    reset_auth_rate_limit()
    body = {"email": "rate@example.com", "password": "pass123456"}
    client.post("/auth/login", json=body)
    response = client.post("/auth/login", json=body)
    assert response.status_code == 429
    assert 1 <= int(response.headers["retry-after"]) <= 60


def test_idle_buckets_are_swept(monkeypatch) -> None:
    """Expired per-client buckets are dropped once the table grows large."""
    from types import SimpleNamespace

    from backend.core import rate_limit

    monkeypatch.setattr(rate_limit, "SWEEP_THRESHOLD", 2)
    reset_auth_rate_limit()
    ticks = iter([0.0, 1.0, 2.0, 1000.0])
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: next(ticks))
    for host in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        rate_limit.enforce_auth_rate_limit(
            SimpleNamespace(client=SimpleNamespace(host=host))
        )
    rate_limit.enforce_auth_rate_limit(
        SimpleNamespace(client=SimpleNamespace(host="10.0.0.4"))
    )
    assert list(rate_limit._hits) == ["10.0.0.4"]
