from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


# ── Auth ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    access_token: str
    refresh_token: str


# ── User ──────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserWithTokens(UserOut):
    access_token: str
    refresh_token: str


# ── Image ─────────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    id: int
    original_name: str
    mime_type: str
    detection_result: Optional[dict] = None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ImageList(BaseModel):
    images: list[ImageOut]


# ── Prediction ────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    boxes: list[list[float]]
    labels: list[int]
    scores: list[float]
    label_names: list[str]
    heatmap: Optional[str] = None  # base64 PNG, only for Faster R-CNN
