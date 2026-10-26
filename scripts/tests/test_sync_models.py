"""Tests for which files the Hugging Face weight sync touches."""

from __future__ import annotations

from pathlib import Path

from scripts.sync_models import local_artifacts, select_artifacts


def test_push_selects_only_weight_files(tmp_path: Path) -> None:
    """Only weights in the artifact dirs are pushed.

    Metadata, backups, git internals, archives, and training outputs stay local.
    """
    for name in [
        "registry/faster_rcnn/v1.1.0/model.safetensors",
        "registry/faster_rcnn/v1.1.0/metadata.json",
        "registry/yolo/v1.0.0/.gitkeep",
        "checkpoints/model.pth",
        "checkpoints/model.pth.backup",
        "checkpoints/.git/lfs/objects/ab/cd/abcd",
        ".weights/swin_unetr/model.pt",
        ".weights/fold0.zip",
        "training/outputs/faster_rcnn/best_model.pth",
    ]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")

    assert local_artifacts(tmp_path) == [
        ".weights/swin_unetr/model.pt",
        "checkpoints/model.pth",
        "registry/faster_rcnn/v1.1.0/model.safetensors",
    ]


def test_pull_never_selects_project_files() -> None:
    """Hub files outside the artifact dirs (model card, LFS config) stay remote."""
    remote = [
        "README.md",
        ".gitattributes",
        "model.pth",
        "registry/yolo/v1.0.0/yolov8_tumor.pt",
    ]
    assert select_artifacts(remote) == ["registry/yolo/v1.0.0/yolov8_tumor.pt"]
