"""Prediction orchestration for Faster R-CNN and YOLOv8."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ml.core.entities import Scan

from ..models import User
from ..pipeline.inference import InferenceEngine, ModelAdapter
from ..schemas.prediction import PredictionResult
from .scan_persistence_service import ScanPersistenceService


class PredictionService:
    """Run inference and persist scan results."""

    def __init__(self, db: Session, adapters: dict[str, ModelAdapter]) -> None:
        """Initialize the service with a database session and injected adapters."""
        self.persistence = ScanPersistenceService(db)
        self.adapters = adapters

    def run(
        self,
        *,
        model_type: str,
        user: User,
        contents: bytes,
        original_name: str | None,
        score_floor: float,
        threshold: float,
        explain: bool = False,
    ) -> PredictionResult:
        """Run inference for a model type, persist, and map to the HTTP schema.

        Args:
            model_type: ``"faster-rcnn"`` or ``"yolov8"``.
            user: Authenticated user instance.
            contents: Raw, already validated image bytes.
            original_name: Original client filename (display only).
            score_floor: Minimum confidence the server returns and stores.
            threshold: Client-visible threshold, persisted with the scan.
            explain: Compute Grad-CAM when True (Faster R-CNN only).
        """
        result = InferenceEngine(self.adapters[model_type]).run(
            contents, score_floor=score_floor, explain=explain
        )
        image = self.persistence.save_scan_result(
            user=user,
            scan=Scan(
                contents=contents,
                original_name=original_name,
                threshold=threshold,
                model_type=result.model_type,
                model_version=result.model_version,
                detections=result.detections,
            ),
        )
        return PredictionResult(
            boxes=result.boxes,
            labels=result.labels,
            scores=result.scores,
            label_names=result.label_names,
            heatmap=result.heatmap,
            model_type=result.model_type,
            model_version=result.model_version,
            image_id=image.id,
        )
