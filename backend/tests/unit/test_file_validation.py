"""Unit tests for upload validation (settings read at call time)."""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from backend.core.exceptions import PayloadTooLargeError, ValidationError
from backend.services import file_validation

_CHUNK = 64 * 1024


class _Upload:
    """Minimal async UploadFile stand-in serving ``data`` in chunks."""

    def __init__(self, data: bytes, content_type: str = "image/png") -> None:
        self.content_type = content_type
        self.reads = 0
        self._buffer = io.BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        self.reads += 1
        return self._buffer.read(size)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point settings at a 1 MiB upload limit."""

    class _Settings:
        max_upload_size_mb = 1

    monkeypatch.setattr(file_validation, "get_settings", lambda: _Settings())


def _read(upload: _Upload) -> bytes:
    return asyncio.run(file_validation.read_image_upload(upload))


def test_accepts_decodable_image() -> None:
    """A real PNG under the cap is returned byte-for-byte."""
    data = _png_bytes()
    assert _read(_Upload(data)) == data


def test_stops_reading_once_cap_exceeded() -> None:
    """Stop streaming once the cap is exceeded; do not drain the file."""
    upload = _Upload(b"x" * _CHUNK * 100)
    with pytest.raises(PayloadTooLargeError):
        _read(upload)
    # 1 MiB / 64 KiB = 16 chunks fit; the 17th exceeds.
    assert upload.reads == 17


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (b"", "image/png"),
        (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml"),
        (b"not an image", "image/jpeg"),
        (_png_bytes(), "text/html"),
    ],
)
def test_rejects_invalid_uploads(data: bytes, content_type: str) -> None:
    """Empty, undecodable, or non-image uploads are validation errors."""
    with pytest.raises(ValidationError):
        _read(_Upload(data, content_type))
