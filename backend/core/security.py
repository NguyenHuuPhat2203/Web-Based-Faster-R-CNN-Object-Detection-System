"""JWT utilities, password hashing, and Google OAuth helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import requests

from ..config import get_settings
from .exceptions import OAuthError, TokenError

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def _make_jti() -> str:
    """Generate a unique JWT identifier."""
    return uuid.uuid4().hex


def create_access_token(user_id: int) -> tuple[str, str, datetime]:
    """Create a signed access token.

    Args:
        user_id: Authenticated user's primary key.

    Returns:
        Tuple of token string, JWT ID, and expiration timestamp.
    """
    jti = _make_jti()
    settings = get_settings()
    exp = utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": str(user_id),
        "jti": jti,
        "type": "access",
        "exp": exp,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token, jti, exp


def create_refresh_token(user_id: int) -> tuple[str, str, datetime]:
    """Create a signed refresh token.

    Args:
        user_id: Authenticated user's primary key.

    Returns:
        Tuple of token string, JWT ID, and expiration timestamp.
    """
    jti = _make_jti()
    settings = get_settings()
    exp = utcnow() + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(user_id),
        "jti": jti,
        "type": "refresh",
        "exp": exp,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token, jti, exp


def decode_token(token: str) -> dict:
    """Decode and validate a JWT.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded token payload.

    Raises:
        TokenError: If the token is expired or invalid.
    """
    try:
        settings = get_settings()
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc


def exchange_google_auth_code(code: str) -> dict:
    """Exchange a Google OAuth authorization code for tokens.

    Args:
        code: One-time authorization code from Google.

    Returns:
        Token response payload from Google.

    Raises:
        OAuthError: If the exchange request fails.
    """
    settings = get_settings()
    data = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code",
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=data, timeout=10)
    if resp.status_code != 200:
        raise OAuthError("Failed to exchange Google auth code")
    return resp.json()


def get_google_user_info(access_token: str) -> dict:
    """Fetch the authenticated Google user's profile.

    Args:
        access_token: Google OAuth access token.

    Returns:
        Google user profile payload.

    Raises:
        OAuthError: If the profile request fails.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(GOOGLE_USERINFO_URL, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise OAuthError("Failed to fetch Google user info")
    return resp.json()
