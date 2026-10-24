"""Pydantic schemas for prediction endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Object-detection inference response (v1 bare payload)."""

    boxes: list[list[float]]
    labels: list[int]
    scores: list[float]
    label_names: list[str]
    heatmap: str | None = None
    model_type: str | None = None
    model_version: str | None = None
    image_id: int | None = None

    model_config = {"from_attributes": True}


class ItemError(BaseModel):
    """Error payload for a single inference item."""

    code: str
    message: str


class InferenceItemOk(BaseModel):
    """Successful v2 inference item."""

    status: Literal["ok"] = "ok"
    result: PredictionResult


class InferenceItemError(BaseModel):
    """Failed v2 inference item."""

    status: Literal["error"] = "error"
    error: ItemError
