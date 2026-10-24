"""Domain entities for detection results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """A single object-detection prediction."""

    box: list[float]
    label: int
    score: float
    label_name: str


@dataclass(frozen=True)
class Scan:
    """An uploaded MRI image plus its detections (the persisted artifact).

    Ownership lives at the persistence seam (``save(user, scan)``), not in
    the artifact itself.
    """

    contents: bytes
    original_name: str | None
    threshold: float
    model_type: str
    model_version: str
    detections: list[Detection]


@dataclass
class InferenceResult:
    """Model inference output before HTTP serialization."""

    boxes: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    heatmap: str | None = None
    model_type: str = ""
    model_version: str = ""

    @property
    def detections(self) -> list[Detection]:
        """Return detections as domain entities."""
        return [
            Detection(box=box, label=label, score=score, label_name=name)
            for box, label, score, name in zip(
                self.boxes, self.labels, self.scores, self.label_names, strict=True
            )
        ]
