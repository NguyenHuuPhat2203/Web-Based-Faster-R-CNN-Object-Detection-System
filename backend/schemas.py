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


# ── Detection ─────────────────────────────────────────────────────────

class DetectionOut(BaseModel):
    label: int
    label_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


# ── Image ─────────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    id: int
    original_name: str
    mime_type: str
    uploaded_at: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    threshold: Optional[float] = None
    detections: list[DetectionOut] = []

    model_config = {"from_attributes": True}


class ImageList(BaseModel):
    images: list[ImageOut]


# ── Prediction ────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    boxes: list[list[float]]
    labels: list[int]
    scores: list[float]
    label_names: list[str]
    heatmap: Optional[str] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None

    model_config = {"from_attributes": True}
