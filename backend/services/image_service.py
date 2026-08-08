"""Image ownership and history business logic."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..core.exceptions import ForbiddenError, NotFoundError
from ..models import Image, User
from ..repositories.image_repository import ImageRepository


class ImageService:
    """Manage user-owned image records."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = ImageRepository(db)

    def list_for_user(self, user: User) -> list[Image]:
        """Return all active images for a user."""
        return self.repo.list_for_user(user.id)

    def get_owned_image(self, user: User, image_id: int) -> Image:
        """Return an image owned by the given user.

        Args:
            user: Authenticated user instance.
            image_id: Target image primary key.

        Returns:
            Owned image instance.

        Raises:
            NotFoundError: If the image does not exist or is deleted.
            ForbiddenError: If the image belongs to another user.
        """
        image = self.repo.get_by_id(image_id)
        if not image or image.deleted:
            raise NotFoundError("Image not found")
        if image.user_id != user.id:
            raise ForbiddenError("Not your image")
        return image

    def soft_delete(self, user: User, image_id: int) -> None:
        """Soft-delete an image owned by the user."""
        image = self.get_owned_image(user, image_id)
        self.repo.soft_delete(image)
