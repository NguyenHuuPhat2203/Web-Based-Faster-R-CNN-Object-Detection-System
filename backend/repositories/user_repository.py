"""Data-access layer for user records."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import User


class UserRepository:
    """Repository for ``User`` persistence operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the repository with a database session."""
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        """Return a user by primary key."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> User | None:
        """Return a user by email address."""
        return self.db.query(User).filter(User.email == email).first()

    def get_by_google_id_or_email(self, google_id: str, email: str) -> User | None:
        """Return a user matching a Google ID or email."""
        return (
            self.db.query(User)
            .filter((User.google_id == google_id) | (User.email == email))
            .first()
        )

    def create(
        self,
        *,
        email: str,
        username: str,
        hashed_password: str | None = None,
        google_id: str | None = None,
    ) -> User:
        """Create and persist a new user."""
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            google_id=google_id,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def save(self, user: User) -> User:
        """Persist changes to an existing user."""
        self.db.commit()
        self.db.refresh(user)
        return user
