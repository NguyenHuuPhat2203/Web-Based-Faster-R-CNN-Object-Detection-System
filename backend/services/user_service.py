"""User profile and password management."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..core import security
from ..core.exceptions import ConflictError, UnauthorizedError, ValidationError
from ..models import User
from ..repositories.user_repository import UserRepository


class UserService:
    """Manage profile updates and password changes."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.users = UserRepository(db)

    def update_profile(
        self,
        current_user: User,
        *,
        username: str | None = None,
        email: str | None = None,
    ) -> User:
        """Update a user's profile fields.

        Args:
            current_user: Authenticated user instance.
            username: Optional new username.
            email: Optional new email address.

        Returns:
            Updated user instance.

        Raises:
            ConflictError: If the email is already in use.
        """
        if email is not None:
            existing = self.users.get_by_email(email)
            if existing and existing.id != current_user.id:
                raise ConflictError("Email already in use")
            current_user.email = email
        if username is not None:
            current_user.username = username
        return self.users.save(current_user)

    def change_password(
        self,
        current_user: User,
        *,
        current_password: str,
        new_password: str,
    ) -> None:
        """Change a local user's password.

        Args:
            current_user: Authenticated user instance.
            current_password: Existing password.
            new_password: Replacement password.

        Raises:
            ValidationError: If the account cannot change passwords.
            UnauthorizedError: If the current password is incorrect.
        """
        if not current_user.hashed_password:
            raise ValidationError(
                "Cannot change password for Google-authenticated accounts"
            )
        if not security.verify_password(current_password, current_user.hashed_password):
            raise UnauthorizedError("Current password is incorrect")
        current_user.hashed_password = security.hash_password(new_password)
        self.users.save(current_user)
