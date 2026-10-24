"""Settings validation for the JWT signing secret."""

import pytest
from pydantic import ValidationError

from backend.config import Settings


def test_rejects_placeholder_and_short_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholder and under-32-character secrets fail validation."""
    monkeypatch.setenv("SECRET_KEY", "change-me-in-production-use-a-real-secret")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("SECRET_KEY", "short")
    with pytest.raises(ValidationError):
        Settings()


def test_accepts_long_random_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """A random secret of at least 32 characters is accepted."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-production-012345")
    assert len(Settings().secret_key) >= 32
