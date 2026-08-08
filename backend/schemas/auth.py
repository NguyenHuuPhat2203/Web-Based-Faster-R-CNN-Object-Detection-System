"""Pydantic schemas for authentication and user endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator


class RegisterRequest(BaseModel):
    """Payload for local user registration."""

    email: str
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        """Validate password length."""
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters")
        return value


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

    username: str | None = None
    email: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        """Validate email format when provided."""
        if value is not None and "@" not in value:
            raise ValueError("Invalid email")
        return value


class PasswordChangeRequest(BaseModel):
    """Payload for password change."""

    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        """Validate new password length."""
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters")
        return value


class UserStats(BaseModel):
    """Aggregated scan statistics for a user."""

    total_scans: int
    total_detections: int
    scans_this_month: int
