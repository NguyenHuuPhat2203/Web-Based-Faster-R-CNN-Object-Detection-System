"""Unit tests for image service ownership checks."""

from backend.core.exceptions import ForbiddenError, NotFoundError
from backend.models import Image, User
from backend.services.image_service import ImageService


class TestImageService:
    """Image service unit tests."""

    def test_get_owned_image_forbidden(self, db_session) -> None:
        """Raise when a user requests another user's image."""
        owner = User(email="owner@example.com", username="Owner")
        other = User(email="other@example.com", username="Other")
        db_session.add(owner)
        db_session.add(other)
        db_session.commit()
        db_session.refresh(owner)
        db_session.refresh(other)

        image = Image(
            user_id=owner.id,
            original_name="scan.png",
            stored_name="scan.png",
            filepath="scan.png",
            mime_type="image/png",
        )
        db_session.add(image)
        db_session.commit()
        db_session.refresh(image)

        service = ImageService(db_session)
        try:
            service.get_owned_image(other, image.id)
            raise AssertionError("Expected ForbiddenError")
        except ForbiddenError:
            pass

    def test_get_owned_image_not_found(self, db_session) -> None:
        """Raise when the image does not exist."""
        user = User(email="solo@example.com", username="Solo")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        service = ImageService(db_session)
        try:
            service.get_owned_image(user, 9999)
            raise AssertionError("Expected NotFoundError")
        except NotFoundError:
            pass
