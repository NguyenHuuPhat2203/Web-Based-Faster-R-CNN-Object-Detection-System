"""Token revocation service."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..core.exceptions import TokenError
from ..core.security import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    decode_token,
)
from ..repositories.token_repository import TokenRepository


class TokenService:
    """Manage JWT blacklist operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.repo = TokenRepository(db)

    def revoke_payload(self, payload: dict) -> bool:
        """Blacklist a decoded token until its natural expiry.

        Args:
            payload: Validated claims from ``decode_token``.

        Returns:
            True if this call revoked the token, False if already revoked.
        """
        expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
        return self.repo.add(payload["jti"], expires_at)

    def revoke_tokens(self, access_token: str, refresh_token: str) -> None:
        """Blacklist the access and refresh tokens that are still valid.

        Tokens that are expired, malformed, or of the wrong type are skipped:
        they cannot be used anyway.

        Args:
            access_token: Raw access token string.
            refresh_token: Raw refresh token string.
        """
        for raw_token, token_type in (
            (access_token, ACCESS_TOKEN_TYPE),
            (refresh_token, REFRESH_TOKEN_TYPE),
        ):
            try:
                payload = decode_token(raw_token, token_type)
            except TokenError:
                continue
            self.revoke_payload(payload)

    def ensure_not_blacklisted(self, jti: str) -> None:
        """Raise when a JWT ID is blacklisted.

        Args:
            jti: JWT identifier from a decoded token.

        Raises:
            TokenError: If the token has been revoked.
        """
        if self.repo.is_blacklisted(jti):
            raise TokenError("Token has been revoked")
