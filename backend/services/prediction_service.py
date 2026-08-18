"""Prediction orchestration for Faster R-CNN and YOLOv8."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ml.core.entities import InferenceResult

from ..models import User
from ..pipeline.adapters import FasterRCNNAdapter, YoloAdapter
from ..pipeline.inference import InferenceEngine
from ..schemas.prediction import PredictionResult
from .scan_persistence_service import ScanPersistenceService

_ADAPTERS = {
    "faster-rcnn": FasterRCNNAdapter(),
    "yolov8": YoloAdapter(),
}


class PredictionService:
    """Run inference and persist scan results."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.persistence = ScanPersistenceService(db)

    @staticmethod
    def _to_response(result: InferenceResult) -> PredictionResult:
        """Map domain inference result to the v1 HTTP schema."""
        return PredictionResult(
            boxes=result.boxes,
            labels=result.labels,
            scores=result.scores,
            label_names=result.label_names,
            heatmap=result.heatmap,
            model_type=result.model_type,
            model_version=result.model_version,
        )

    def run(
        self,
        *,
        model_type: str,
        user: User,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        score_floor: float,
        threshold: float | None = None,
        explain: bool = False,
    ) -> PredictionResult:
        """Run inference for a model type, persist, and map to the HTTP schema.

        Args:
            model_type: ``"faster-rcnn"`` or ``"yolov8"``.
            user: Authenticated user instance.
            contents: Raw image bytes.
            original_name: Original client filename.
            content_type: MIME type reported by the client.
            score_floor: Minimum confidence the server returns.
            threshold: Client-visible threshold, persisted as metadata only.
            explain: Compute Grad-CAM when True (Faster R-CNN only).
        """
        adapter = _ADAPTERS[model_type]
        result = InferenceEngine(adapter).run(
            contents, score_floor=score_floor, explain=explain
        )
        self._persist(
            user=user,
            contents=contents,
            original_name=original_name,
            content_type=content_type,
            threshold=threshold if threshold is not None else score_floor,
            result=result,
        )
        return self._to_response(result)

    def _persist(
        self,
        *,
        user: User,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
        result: InferenceResult,
    ) -> None:
        """Persist scan detections for the authenticated user."""
        self.persistence.save_scan_result(
            user=user,
            contents=contents,
            original_name=original_name,
            content_type=content_type,
            threshold=threshold,
            model_type=result.model_type,
            model_version=result.model_version,
            detections=result.detections,
        )
