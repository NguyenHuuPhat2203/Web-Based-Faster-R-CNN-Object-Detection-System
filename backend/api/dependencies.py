"""FastAPI dependency-injection helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.exceptions import UnauthorizedError
from ..core.security import ACCESS_TOKEN_TYPE, decode_token
from ..models import User
from ..repositories.user_repository import UserRepository
from ..services.token_service import TokenService

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Extract and validate the access JWT and return the authenticated user.

    Args:
        credentials: Bearer token from the Authorization header.
        db: Database session.

    Returns:
        Authenticated user instance.

    Raises:
        UnauthorizedError: If the header is missing or the user no longer exists.
        TokenError: If the token is invalid, not an access token, or revoked.
    """
    if credentials is None:
        raise UnauthorizedError("Missing authorization header")

    payload = decode_token(credentials.credentials, ACCESS_TOKEN_TYPE)
    TokenService(db).ensure_not_blacklisted(payload["jti"])

    user = UserRepository(db).get_by_id(int(payload["sub"]))
    if user is None:
        raise UnauthorizedError("User not found")
    return user
