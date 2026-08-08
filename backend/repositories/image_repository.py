"""Data-access layer for image and detection records."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import Detection, Image


class ImageRepository:
    """Repository for ``Image`` and ``Detection`` persistence operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the repository with a database session."""
        self.db = db

    def list_for_user(self, user_id: int) -> list[Image]:
        """Return non-deleted images for a user ordered by upload time."""
        return (
            self.db.query(Image)
            .filter(Image.user_id == user_id, ~Image.deleted)
            .order_by(Image.uploaded_at.desc())
            .all()
        )

    def get_by_id(self, image_id: int) -> Image | None:
        """Return an image by primary key."""
        return self.db.query(Image).filter(Image.id == image_id).first()

    def soft_delete(self, image: Image) -> None:
        """Mark an image as deleted."""
        image.deleted = True
        self.db.commit()

    def create_with_detections(
        self,
        image: Image,
        detections: list[Detection],
    ) -> Image:
        """Persist an image and its detections."""
        self.db.add(image)
        self.db.flush()
        for detection in detections:
            detection.image_id = image.id
            self.db.add(detection)
        self.db.commit()
        self.db.refresh(image)
        return image
