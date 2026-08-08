"""Shared Pydantic configs for model and data (no secrets)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt


class DataConfig(BaseModel):
    """Dataset and dataloader configuration."""

    train_path: Path = Path("data/raw/v1/train")
    val_path: Path = Path("data/raw/v1/valid")
    dataset_version: str = "v1"
    batch_size: PositiveInt = Field(default=4, description="Per-device batch size")
    num_workers: int = Field(default=2, ge=0)
    image_size: PositiveInt = Field(default=640)


class ModelConfig(BaseModel):
    """Architecture configuration shared by train and serve."""

    name: str = "faster_rcnn_resnet50_fpn"
    num_classes: PositiveInt = 3
    score_threshold: NonNegativeFloat = Field(default=0.5, le=1.0)
    version: str = "resnet50-fpn-v1"


class TrainConfig(BaseModel):
    """Training hyperparameters."""

    learning_rate: NonNegativeFloat = Field(default=0.005)
    momentum: NonNegativeFloat = Field(default=0.9)
    weight_decay: NonNegativeFloat = Field(default=0.0005)
    epochs: PositiveInt = Field(default=10)
    seed: int = 42
    gradient_accumulation_steps: PositiveInt = 1
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
