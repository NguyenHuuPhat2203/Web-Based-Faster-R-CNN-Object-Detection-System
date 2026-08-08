"""Prediction orchestration for Faster R-CNN and YOLOv8."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ml.core.entities import InferenceResult

from ..models import User
from ..pipeline import run_faster_rcnn, run_yolo
from ..schemas.prediction import PredictionResult
from .scan_persistence_service import (
    DetectionRecord,
    ScanPersistenceService,
)


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

    def run_faster_rcnn(
        self,
        *,
        user: User,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
        explain: bool = True,
    ) -> PredictionResult:
        """Run Faster R-CNN inference on an uploaded image.

        Args:
            user: Authenticated user instance.
            contents: Raw image bytes.
            original_name: Original client filename.
            content_type: MIME type reported by the client.
            threshold: Confidence threshold for detections.
            explain: Compute Grad-CAM when True (v1 default).

        Returns:
            Prediction response including optional Grad-CAM heatmap.
        """
        pil_image, result = run_faster_rcnn(
            contents, threshold=threshold, explain=explain
        )
        self._persist(
            user=user,
            pil_image=pil_image,
            contents=contents,
            original_name=original_name,
            content_type=content_type,
            threshold=threshold,
            result=result,
        )
        return self._to_response(result)

    def run_yolo(
        self,
        *,
        user: User,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
    ) -> PredictionResult:
        """Run YOLOv8 inference on an uploaded image.

        Args:
            user: Authenticated user instance.
            contents: Raw image bytes.
            original_name: Original client filename.
            content_type: MIME type reported by the client.
            threshold: Confidence threshold for detections.

        Returns:
            Prediction response without heatmap data.
        """
        pil_image, result = run_yolo(contents, threshold=threshold)
        self._persist(
            user=user,
            pil_image=pil_image,
            contents=contents,
            original_name=original_name,
            content_type=content_type,
            threshold=threshold,
            result=result,
        )
        return self._to_response(result)

    def _persist(
        self,
        *,
        user: User,
        pil_image,
        contents: bytes,
        original_name: str | None,
        content_type: str | None,
        threshold: float,
        result: InferenceResult,
    ) -> None:
        """Persist scan detections for the authenticated user."""
        detection_records = [
            DetectionRecord(
                box=det.box,
                label=det.label,
                score=det.score,
                label_name=det.label_name,
            )
            for det in result.detections
        ]
        self.persistence.save_scan_result(
            user=user,
            pil_image=pil_image,
            contents=contents,
            original_name=original_name,
            content_type=content_type,
            threshold=threshold,
            model_type=result.model_type,
            model_version=result.model_version,
            detections=detection_records,
        )
