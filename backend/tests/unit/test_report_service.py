"""Unit tests for PDF report content helpers."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from types import SimpleNamespace

from PIL import Image as PILImage

from backend.services.report_service import (
    annotated_preview_bytes,
    meta_rows,
    reported_detections,
)


def _black_png(path, size: int = 32) -> None:
    PILImage.new("RGB", (size, size), color=(0, 0, 0)).save(path)


def test_annotated_preview_draws_box_outline(tmp_path) -> None:
    """Box outline pixels differ from the original black background."""
    path = tmp_path / "scan.png"
    _black_png(path)
    image = SimpleNamespace(
        filepath=str(path),
        threshold=0.5,
        detections=[
            SimpleNamespace(
                label_name="positive", score=0.9, x1=5.0, y1=5.0, x2=20.0, y2=20.0
            )
        ],
    )

    painted = PILImage.open(io.BytesIO(annotated_preview_bytes(image))).convert("RGB")
    assert painted.getpixel((5, 5)) != (0, 0, 0)


def test_meta_rows_report_only_detections_above_threshold() -> None:
    """Metadata counts and averages the detections the user saw."""
    image = SimpleNamespace(
        id=7,
        original_name="scan.png",
        uploaded_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        model_type="faster-rcnn",
        model_version="v1.0.0",
        threshold=0.5,
        detections=[
            SimpleNamespace(score=0.8),
            SimpleNamespace(score=0.6),
            SimpleNamespace(score=0.2),  # stored by v2 floor, not reported
        ],
    )
    rows = dict(meta_rows(image))
    assert rows["Scan ID"] == "IMG-000007"
    assert rows["Model"] == "faster-rcnn"
    assert rows["Version"] == "v1.0.0"
    assert rows["Threshold"] == "0.50"
    assert rows["Detections"] == "2"
    assert rows["Avg Confidence"] == "70.0%"


def test_reported_detections_without_threshold_keeps_all() -> None:
    """Legacy rows without a stored threshold report every detection."""
    image = SimpleNamespace(
        threshold=None,
        detections=[SimpleNamespace(score=0.9), SimpleNamespace(score=0.3)],
    )
    assert len(reported_detections(image)) == 2
