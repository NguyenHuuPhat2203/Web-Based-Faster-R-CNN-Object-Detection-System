"""Data-access layer for user records."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.exceptions import ConflictError
from ..models import User


class UserRepository:
    """Repository for ``User`` persistence operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the repository with a database session."""
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        """Return a user by primary key."""
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """Return a user by email address, ignoring case."""
        return (
            self.db.query(User)
            .filter(func.lower(User.email) == email.strip().lower())
            .first()
        )

    def get_by_google_id(self, google_id: str) -> User | None:
        """Return a user linked to a Google account ID."""
        return self.db.query(User).filter(User.google_id == google_id).first()

    def create(
        self,
        *,
        email: str,
        username: str,
        hashed_password: str | None = None,
        google_id: str | None = None,
    ) -> User:
        """Create and persist a new user.

        Raises:
            ConflictError: If the email or Google ID is already taken.
        """
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            google_id=google_id,
        )
        self.db.add(user)
        return self.save(user)

    def save(self, user: User) -> User:
        """Persist changes to a user.

        Raises:
            ConflictError: If a unique constraint (email, Google ID) is violated.
        """
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raise ConflictError("Email or Google account already in use") from None
        self.db.refresh(user)
        return user
