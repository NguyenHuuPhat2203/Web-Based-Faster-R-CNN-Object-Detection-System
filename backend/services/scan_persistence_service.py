"""Persist scan uploads and detection results."""

from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy.orm import Session

from ml.core.entities import Scan

from ..config import get_settings
from ..models import Detection as DetectionRow
from ..models import Image, User
from ..repositories.image_repository import ImageRepository


class ScanPersistenceService:
    """Save uploaded images and detection rows for a user."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = ImageRepository(db)

    def save_scan_result(self, *, user: User, scan: Scan) -> Image:
        """Persist an uploaded image (raw bytes) and its detections.

        MIME type and file extension come from the decoded image format, never
        from the client-supplied name or ``Content-Type``, because the stored
        file is later served back with that MIME type.

        Args:
            user: Authenticated user instance.
            scan: The scan artifact to persist (bytes already validated).

        Returns:
            Persisted image ORM instance.
        """
        with PILImage.open(io.BytesIO(scan.contents)) as decoded:
            image_format = decoded.format or "PNG"
            # Same frame the detector used (see InferenceEngine decode).
            width, height = ImageOps.exif_transpose(decoded).size
        mime_type = PILImage.MIME.get(image_format, "application/octet-stream")
        extension = f".{image_format.lower()}"

        user_dir = Path(get_settings().upload_dir) / str(user.id)
        user_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}{extension}"
        file_path = user_dir / stored_name
        file_path.write_bytes(scan.contents)

        image = Image(
            user_id=user.id,
            original_name=scan.original_name or "unknown",
            stored_name=stored_name,
            filepath=str(file_path),
            mime_type=mime_type,
            width=width,
            height=height,
            file_size=len(scan.contents),
            content_hash=hashlib.sha256(scan.contents).hexdigest(),
            model_type=scan.model_type,
            model_version=scan.model_version,
            threshold=scan.threshold,
        )
        detection_rows = [
            DetectionRow(
                label=detection.label,
                label_name=detection.label_name,
                score=detection.score,
                x1=detection.box[0],
                y1=detection.box[1],
                x2=detection.box[2],
                y2=detection.box[3],
            )
            for detection in scan.detections
        ]
        return self.repo.create_with_detections(image, detection_rows)
