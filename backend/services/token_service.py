"""Token revocation service."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..core.exceptions import TokenError
from ..core.security import decode_token, utcnow
from ..repositories.token_repository import TokenRepository


class TokenService:
    """Manage JWT blacklist operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = TokenRepository(db)

    def revoke_tokens(self, access_token: str, refresh_token: str) -> None:
        """Blacklist access and refresh tokens when possible.

        Args:
            access_token: Raw access token string.
            refresh_token: Raw refresh token string.
        """
        now = utcnow()
        for raw_token in (access_token, refresh_token):
            try:
                payload = decode_token(raw_token)
            except TokenError:
                continue

            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti:
                expires_at = datetime.fromtimestamp(exp, tz=UTC) if exp else now
                self.repo.add(jti, expires_at)

    def ensure_not_blacklisted(self, jti: str | None) -> None:
        """Raise when a JWT ID is blacklisted.

        Args:
            jti: JWT identifier from a decoded token.

        Raises:
            TokenError: If the token has been revoked.
        """
        if jti and self.repo.is_blacklisted(jti):
            raise TokenError("Token has been revoked")
