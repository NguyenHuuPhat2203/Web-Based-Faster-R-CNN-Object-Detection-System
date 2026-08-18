"""Persist scan uploads and detection results."""

from __future__ import annotations

import hashlib
import io
import os
import uuid

from PIL import Image as PILImage
from sqlalchemy.orm import Session

from ml.core.entities import Detection

from ..config import get_settings
from ..models import Detection as DetectionRow
from ..models import Image, User
from ..repositories.image_repository import ImageRepository

settings = get_settings()


def _image_dimensions(contents: bytes) -> tuple[int | None, int | None]:
    """Return (width, height) from the image header without a full decode."""
    try:
        with PILImage.open(io.BytesIO(contents)) as img:
            return img.width, img.height
    except Exception:
        return None, None


class ScanPersistenceService:
    """Save uploaded images and detection rows for a user."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = ImageRepository(db)

    def save_scan_result(
        self,
        *,
        user: User,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
        model_type: str,
        model_version: str,
        detections: list[Detection],
    ) -> Image:
        """Persist an uploaded image (raw bytes) and its detections.

        Args:
            user: Authenticated user instance.
            contents: Raw upload bytes.
            original_name: Original client filename.
            content_type: MIME type reported by the client.
            threshold: Confidence threshold used for inference.
            model_type: Model identifier string.
            model_version: Model version string.
            detections: Domain detection entities to persist.

        Returns:
            Persisted image ORM instance.
        """
        user_dir = os.path.join(settings.upload_dir, str(user.id))
        os.makedirs(user_dir, exist_ok=True)

        extension = os.path.splitext(original_name or "image.jpg")[1] or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(user_dir, stored_name)
        with open(file_path, "wb") as fh:
            fh.write(contents)

        width, height = _image_dimensions(contents)

        image = Image(
            user_id=user.id,
            original_name=original_name or "unknown",
            stored_name=stored_name,
            filepath=file_path,
            mime_type=content_type or "image/jpeg",
            width=width,
            height=height,
            file_size=len(contents),
            content_hash=hashlib.sha256(contents).hexdigest(),
            model_type=model_type,
            model_version=model_version,
            threshold=threshold,
        )

        detection_rows = [
            DetectionRow(
                label=det.label,
                label_name=det.label_name,
                score=det.score,
                x1=det.box[0],
                y1=det.box[1],
                x2=det.box[2],
                y2=det.box[3],
            )
            for det in detections
        ]
        return self.repo.create_with_detections(image, detection_rows)
