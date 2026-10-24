"""JWT utilities, password hashing, and Google OAuth helpers."""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import requests

from ..config import get_settings
from .exceptions import OAuthError, TokenError

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"
# bcrypt only hashes the first 72 bytes; newer releases raise on longer input.
BCRYPT_MAX_PASSWORD_BYTES = 72
_REQUIRED_TOKEN_CLAIMS = ["exp", "sub", "jti", "type"]


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt.

    Raises:
        ValueError: If the password exceeds the bcrypt 72-byte limit.
    """
    encoded = password.encode()
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError("Password exceeds 72 bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    encoded = plain.encode()
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode())
    except ValueError:
        return False


@functools.cache
def _dummy_password_hash() -> str:
    """Return a throwaway bcrypt hash used to equalize login timing."""
    return hash_password(secrets.token_urlsafe(16))


def burn_password_check(plain: str) -> None:
    """Spend one bcrypt verification so unknown accounts cost the same as known."""
    verify_password(plain, _dummy_password_hash())


def utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def _make_jti() -> str:
    """Generate a unique JWT identifier."""
    return uuid.uuid4().hex


def _create_token(user_id: int, token_type: str, lifetime: timedelta) -> str:
    """Sign a JWT of the given type with a fresh unique ``jti``."""
    settings = get_settings()
    payload = {
        "sub": str(user_id),
        "jti": _make_jti(),
        "type": token_type,
        "exp": utcnow() + lifetime,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_access_token(user_id: int) -> str:
    """Create a signed short-lived access token for a user."""
    minutes = get_settings().access_token_expire_minutes
    return _create_token(user_id, ACCESS_TOKEN_TYPE, timedelta(minutes=minutes))


def create_refresh_token(user_id: int) -> str:
    """Create a signed long-lived refresh token for a user."""
    days = get_settings().refresh_token_expire_days
    return _create_token(user_id, REFRESH_TOKEN_TYPE, timedelta(days=days))


def decode_token(token: str, expected_type: str | None = None) -> dict:
    """Decode and validate a JWT.

    Args:
        token: Encoded JWT string.
        expected_type: Required ``type`` claim (access or refresh), or None to
            accept any type.

    Returns:
        Decoded token payload.

    Raises:
        TokenError: If the token is expired, malformed, lacks required claims, or
            has the wrong type.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"require": _REQUIRED_TOKEN_CLAIMS},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc
    if expected_type is not None and payload["type"] != expected_type:
        raise TokenError(f"Invalid token type, expected {expected_type} token")
    return payload


def create_pkce_pair() -> tuple[str, str]:
    """Return a PKCE verifier and its S256 challenge."""
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def dump_oauth_tx(state: str, code_verifier: str) -> str:
    """Sign a 10-minute OAuth transaction cookie."""
    payload = json.dumps(
        {"state": state, "v": code_verifier, "exp": int(time.time()) + 600}
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).decode()
    digest = hmac.new(
        get_settings().secret_key.encode(),
        encoded.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{digest}"


def load_oauth_tx(token: str) -> tuple[str, str]:
    """Validate a signed OAuth transaction cookie."""
    try:
        encoded, digest = token.split(".", 1)
        expected = hmac.new(
            get_settings().secret_key.encode(),
            encoded.encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, digest):
            raise OAuthError("Invalid OAuth state")
        data = json.loads(base64.urlsafe_b64decode(encoded))
        state = data["state"]
        verifier = data["v"]
        exp = data["exp"]
    except (OAuthError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise OAuthError("Invalid OAuth state") from None
    if exp <= time.time():
        raise OAuthError("Invalid OAuth state")
    return state, verifier


def _google_request(method: str, url: str, error: str, **kwargs) -> dict:
    """Call a Google endpoint and return its JSON object payload.

    Raises:
        OAuthError: With ``error`` on network failure, non-200, or non-object JSON.
    """
    try:
        resp = requests.request(method, url, timeout=10, **kwargs)
        payload = resp.json() if resp.status_code == 200 else None
    except (requests.RequestException, ValueError):
        raise OAuthError(error) from None
    if not isinstance(payload, dict):
        raise OAuthError(error)
    return payload


def exchange_google_auth_code(code: str, code_verifier: str) -> dict:
    """Exchange a Google OAuth authorization code for tokens.

    Args:
        code: One-time authorization code from Google.
        code_verifier: PKCE verifier bound to the authorization request.

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
        "code_verifier": code_verifier,
    }
    return _google_request(
        "POST", GOOGLE_TOKEN_URL, "Failed to exchange Google auth code", data=data
    )


def get_google_user_info(access_token: str) -> dict:
    """Fetch the authenticated Google user's profile.

    Args:
        access_token: Google OAuth access token.

    Returns:
        Google user profile payload.

    Raises:
        OAuthError: If the profile request fails.
    """
    return _google_request(
        "GET",
        GOOGLE_USERINFO_URL,
        "Failed to fetch Google user info",
        headers={"Authorization": f"Bearer {access_token}"},
    )
