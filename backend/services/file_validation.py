"""Upload validation: size cap, MIME family, and decodable image content."""

from __future__ import annotations

import io

from fastapi import UploadFile
from PIL import Image as PILImage

from ..config import get_settings
from ..core.exceptions import PayloadTooLargeError, ValidationError

_CHUNK_BYTES = 64 * 1024


async def read_image_upload(file: UploadFile) -> bytes:
    """Read an image upload in 64 KiB chunks, enforcing every upload rule.

    Args:
        file: Incoming FastAPI upload.

    Returns:
        Raw file bytes.

    Raises:
        ValidationError: If the declared type is not ``image/*``, the upload is
            empty, or the bytes are not an image Pillow can decode.
        PayloadTooLargeError: If the stream exceeds ``max_upload_size_mb``.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValidationError("File must be an image")

    max_upload_size_mb = get_settings().max_upload_size_mb
    max_bytes = max_upload_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(f"File exceeds {max_upload_size_mb} MB limit")
        chunks.append(chunk)
    if total == 0:
        raise ValidationError("Empty upload")

    contents = b"".join(chunks)
    try:
        with PILImage.open(io.BytesIO(contents)) as image:
            image.verify()
    except (OSError, SyntaxError, PILImage.DecompressionBombError) as exc:
        raise ValidationError("File is not a valid image") from exc
    return contents
