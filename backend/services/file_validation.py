"""Upload validation helpers."""

from __future__ import annotations

from ..config import get_settings
from ..core.exceptions import PayloadTooLargeError, ValidationError

settings = get_settings()


def validate_image_upload(
    *,
    content_type: str | None,
    size_bytes: int,
) -> None:
    """Validate uploaded image metadata and size.

    Args:
        content_type: MIME type reported by the client.
        size_bytes: Raw upload size in bytes.

    Raises:
        ValidationError: If the upload is not a valid image or is too large.
    """
    if not content_type or not content_type.startswith("image/"):
        raise ValidationError("File must be an image")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise PayloadTooLargeError(
            f"File exceeds {settings.max_upload_size_mb} MB limit"
        )
