"""Persist scan uploads and detection results."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass

from PIL import Image as PILImage
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Detection, Image, User
from ..repositories.image_repository import ImageRepository

settings = get_settings()


@dataclass
class DetectionRecord:
    """Serializable detection result used for persistence."""

    box: list[float]
    label: int
    score: float
    label_name: str


class ScanPersistenceService:
    """Save uploaded images and detection rows for a user."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = ImageRepository(db)

    def save_scan_result(
        self,
        *,
        user: User,
        pil_image: PILImage.Image,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
        model_type: str,
        model_version: str,
        detections: list[DetectionRecord],
    ) -> Image:
        """Persist an uploaded image and its detections.

        Args:
            user: Authenticated user instance.
            pil_image: Opened RGB image.
            contents: Raw upload bytes.
            original_name: Original client filename.
            content_type: MIME type reported by the client.
            threshold: Confidence threshold used for inference.
            model_type: Model identifier string.
            model_version: Model version string.
            detections: Detection records to persist.

        Returns:
            Persisted image ORM instance.
        """
        user_dir = os.path.join(settings.upload_dir, str(user.id))
        os.makedirs(user_dir, exist_ok=True)

        extension = os.path.splitext(original_name or "image.jpg")[1] or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(user_dir, stored_name)
        pil_image.save(file_path)

        image = Image(
            user_id=user.id,
            original_name=original_name or "unknown",
            stored_name=stored_name,
            filepath=file_path,
            mime_type=content_type or "image/jpeg",
            width=pil_image.width,
            height=pil_image.height,
            file_size=len(contents),
            content_hash=hashlib.sha256(contents).hexdigest(),
            model_type=model_type,
            model_version=model_version,
            threshold=threshold,
        )

        detection_rows = [
            Detection(
                label=record.label,
                label_name=record.label_name,
                score=record.score,
                x1=record.box[0],
                y1=record.box[1],
                x2=record.box[2],
                y2=record.box[3],
            )
            for record in detections
        ]
        return self.repo.create_with_detections(image, detection_rows)
