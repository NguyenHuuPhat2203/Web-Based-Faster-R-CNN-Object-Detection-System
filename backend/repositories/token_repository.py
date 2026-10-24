"""Data-access layer for revoked JWT records."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.security import utcnow
from ..models import TokenBlacklist


class TokenRepository:
    """Repository for token blacklist persistence operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the repository with a database session."""
        self.db = db

    def is_blacklisted(self, jti: str) -> bool:
        """Return whether a JWT ID is blacklisted."""
        return (
            self.db.query(TokenBlacklist.id).filter(TokenBlacklist.jti == jti).first()
            is not None
        )

    def add(self, jti: str, expires_at: datetime) -> bool:
        """Blacklist a JWT ID atomically.

        The unique index on ``jti`` makes concurrent revocations of the same
        token race-free: exactly one caller gets ``True``.

        Returns:
            True if this call revoked the token, False if it was already revoked.
        """
        self.db.add(TokenBlacklist(jti=jti, expires_at=expires_at))
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            return False
        return True

    def purge_expired(self) -> int:
        """Delete expired blacklist entries and return the removed count."""
        count = (
            self.db.query(TokenBlacklist)
            .filter(TokenBlacklist.expires_at < utcnow())
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return count
