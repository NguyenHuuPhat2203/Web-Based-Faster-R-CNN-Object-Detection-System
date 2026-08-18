"""Unit tests for upload validation (settings read at call time)."""

from __future__ import annotations

import pytest

from backend.core.exceptions import PayloadTooLargeError
from backend.services import file_validation


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point settings at a small configurable upload limit."""

    class _Settings:
        max_upload_size_mb = 1

    monkeypatch.setattr(file_validation, "get_settings", lambda: _Settings())


def test_validates_size_against_current_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload limit is read per call, not snapshotted at import."""

    class _Big:
        max_upload_size_mb = 10

    monkeypatch.setattr(file_validation, "get_settings", lambda: _Big())
    file_validation.validate_image_upload(
        content_type="image/png", size_bytes=2 * 1024 * 1024
    )


def test_rejects_oversize_upload() -> None:
    """Files larger than the configured limit are rejected."""
    with pytest.raises(PayloadTooLargeError):
        file_validation.validate_image_upload(
            content_type="image/jpeg", size_bytes=2 * 1024 * 1024
        )
