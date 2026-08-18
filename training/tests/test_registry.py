"""Unit tests for the registry writer seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.scripts.registry import RegistryGuardError, write_registry_entry


def _write_weights(tmp_path: Path) -> Path:
    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    return src


def test_write_entry_copies_weights_and_metadata(tmp_path) -> None:
    """A plain entry copies weights as-is and writes a shaped metadata.json."""
    src = _write_weights(tmp_path)
    registry = tmp_path / "registry"

    out = write_registry_entry(
        model_key="yolo",
        version="v0.0.1",
        dataset_version="v1",
        dataset_hash="abc",
        metrics={"mAP@0.5": 0.7},
        weights_src=src,
        weights_name="yolov8_tumor.pt",
        registry_root=registry,
    )

    assert out.read_bytes() == b"fake-weights"
    meta = json.loads((registry / "yolo" / "v0.0.1" / "metadata.json").read_text())
    assert meta["model_type"] == "yolo"
    assert meta["version"] == "v0.0.1"
    assert meta["dataset_hash"] == "abc"
    assert meta["metrics"] == {"mAP@0.5": 0.7}
    assert meta["weights"] == "yolov8_tumor.pt"


def test_write_entry_applies_converter(tmp_path) -> None:
    """A converter callback transforms weights before they land in the registry."""
    src = _write_weights(tmp_path)
    registry = tmp_path / "registry"
    converted = []

    def convert(s: Path, d: Path) -> None:
        converted.append((s.name, d.name))
        d.write_bytes(b"converted")

    out = write_registry_entry(
        model_key="faster_rcnn",
        version="v0.0.1",
        dataset_version="v1",
        dataset_hash="abc",
        metrics={"mAP@0.5": 0.8},
        weights_src=src,
        weights_name="model.safetensors",
        convert=convert,
        registry_root=registry,
    )

    assert out.read_bytes() == b"converted"
    assert converted == [("weights.pt", "model.safetensors")]


def test_production_version_requires_i_know(tmp_path) -> None:
    """v1.0.0 is refused without the explicit confirmation."""
    src = _write_weights(tmp_path)
    with pytest.raises(RegistryGuardError):
        write_registry_entry(
            model_key="yolo",
            version="v1.0.0",
            dataset_version="v1",
            dataset_hash="",
            metrics={},
            weights_src=src,
            weights_name="yolov8_tumor.pt",
            registry_root=tmp_path / "registry",
        )


def test_production_version_allowed_with_i_know(tmp_path) -> None:
    """v1.0.0 passes once --i-know is given."""
    src = _write_weights(tmp_path)
    out = write_registry_entry(
        model_key="yolo",
        version="v1.0.0",
        dataset_version="v1",
        dataset_hash="",
        metrics={},
        weights_src=src,
        weights_name="yolov8_tumor.pt",
        registry_root=tmp_path / "registry",
        i_know=True,
    )
    assert out.exists()
