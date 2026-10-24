"""Pydantic schemas for authentication and user endpoints."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel

from ..core.security import BCRYPT_MAX_PASSWORD_BYTES

MIN_PASSWORD_LENGTH = 6
MAX_EMAIL_LENGTH = 255
MAX_USERNAME_LENGTH = 100
_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+")


def _normalize_email(value: str) -> str:
    """Trim, lowercase, and sanity-check an email address."""
    email = value.strip().lower()
    if len(email) > MAX_EMAIL_LENGTH or not _EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Invalid email")
    return email


def _normalize_username(value: str) -> str:
    """Trim a username and enforce the column length."""
    username = value.strip()
    if not username or len(username) > MAX_USERNAME_LENGTH:
        raise ValueError(f"Username must be 1-{MAX_USERNAME_LENGTH} characters")
    return username


def _check_password_strength(value: str) -> str:
    """Enforce minimum length and the bcrypt 72-byte ceiling."""
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(value.encode()) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {BCRYPT_MAX_PASSWORD_BYTES} bytes")
    return value


Email = Annotated[str, AfterValidator(_normalize_email)]
Username = Annotated[str, AfterValidator(_normalize_username)]
NewPassword = Annotated[str, AfterValidator(_check_password_strength)]


class RegisterRequest(BaseModel):
    """Payload for local user registration."""

    email: Email
    username: Username
    password: NewPassword


class LoginRequest(BaseModel):
    """Payload for local user login."""

    email: str
    password: str


class TokenResponse(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Payload for token refresh."""

    refresh_token: str


class LogoutRequest(BaseModel):
    """Payload for logout and token revocation."""

    access_token: str
    refresh_token: str


class UserOut(BaseModel):
    """Public user profile fields."""

    id: int
    email: str
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserWithTokens(UserOut):
    """User profile returned alongside issued tokens."""

    access_token: str
    refresh_token: str


class ProfileUpdateRequest(BaseModel):
    """Payload for profile updates."""

    username: Username | None = None
    email: Email | None = None


class PasswordChangeRequest(BaseModel):
    """Payload for password change."""

    current_password: str
    new_password: NewPassword


class UserStats(BaseModel):
    """Aggregated scan statistics for a user."""

    total_scans: int
    total_detections: int
    scans_this_month: int
