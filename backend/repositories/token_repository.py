"""Data-access layer for revoked JWT records."""

from __future__ import annotations

from datetime import datetime

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
            self.db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
            is not None
        )

    def add(self, jti: str, expires_at: datetime) -> None:
        """Add a JWT ID to the blacklist if it is not already present."""
        exists = self.db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
        if not exists:
            self.db.add(TokenBlacklist(jti=jti, expires_at=expires_at))
            self.db.commit()

    def purge_expired(self) -> int:
        """Delete expired blacklist entries and return the removed count."""
        count = (
            self.db.query(TokenBlacklist)
            .filter(TokenBlacklist.expires_at < utcnow())
            .delete(synchronize_session="fetch")
        )
        if count:
            self.db.commit()
        return count
