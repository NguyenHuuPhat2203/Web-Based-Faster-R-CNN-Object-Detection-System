"""Pydantic schemas for image and detection endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DetectionOut(BaseModel):
    """Serialized detection result."""

    label: int
    label_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


class ImageOut(BaseModel):
    """Serialized uploaded image with detections."""

    id: int
    original_name: str
    mime_type: str
    uploaded_at: datetime
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    model_type: str | None = None
    model_version: str | None = None
    threshold: float | None = None
    detections: list[DetectionOut] = []

    model_config = {"from_attributes": True}


class ImageList(BaseModel):
    """List wrapper for image history responses."""

    images: list[ImageOut]
