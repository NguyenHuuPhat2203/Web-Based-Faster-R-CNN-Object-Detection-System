"""Shared Pydantic configs for model and data (no secrets)."""

from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from ml.core.constants import (
    DEFAULT_SCORE_THRESHOLD,
    FASTER_RCNN_MODEL_VERSION,
    FASTER_RCNN_NUM_CLASSES,
)


class DataConfig(BaseModel):
    """Dataloader configuration."""

    batch_size: PositiveInt = Field(default=4, description="Per-device batch size")
    num_workers: int = Field(default=0, ge=0)


class ModelConfig(BaseModel):
    """Architecture configuration shared by train and serve."""

    name: str = "faster_rcnn_resnet50_fpn"
    num_classes: PositiveInt = FASTER_RCNN_NUM_CLASSES
    score_threshold: float = Field(default=DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    version: str = FASTER_RCNN_MODEL_VERSION


class TrainConfig(BaseModel):
    """Training hyperparameters."""

    learning_rate: PositiveFloat = 0.005
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    epochs: PositiveInt = 10
    seed: int = 42
    gradient_accumulation_steps: PositiveInt = 1
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
