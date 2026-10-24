"""Application configuration loaded from environment variables."""

import re
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


def resolve_database_url(url: str) -> str:
    """Anchor relative SQLite file paths to the backend package directory.

    After the serve layout change, the process cwd is the repo root. A URL like
    ``sqlite:///./brain_tumor.db`` would otherwise open/create an empty DB at the
    repo root instead of the migrated ``backend/brain_tumor.db``.
    """
    if not url.startswith("sqlite"):
        return url
    database = make_url(url).database
    if not database:
        return url
    path = Path(database)
    if path.is_absolute():
        return url
    resolved = (_BACKEND_DIR / path).resolve()
    return f"sqlite:///{resolved.as_posix()}"


_PLACEHOLDER_SECRETS = frozenset(
    {
        "change-me-in-production-use-a-real-secret",
        "change-me-to-a-long-random-secret",
    }
)
_REGISTRY_VERSION_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+){1,2}(?:-[A-Za-z0-9.]+)?$")


class Settings(BaseSettings):
    """Runtime settings for the Brain Tumor Detector API.

    Attributes:
        secret_key: JWT signing secret.
        algorithm: JWT signing algorithm.
        access_token_expire_minutes: Access token lifetime in minutes.
        refresh_token_expire_days: Refresh token lifetime in days.
        google_client_id: Google OAuth client ID.
        google_client_secret: Google OAuth client secret.
        google_redirect_uri: Google OAuth callback URL.
        database_url: SQLAlchemy database connection string.
        upload_dir: Directory for persisted user uploads.
        max_upload_size_mb: Maximum upload size in megabytes.
        frontend_url: Frontend origin used for CORS and OAuth redirects.
        case_viewer_dir: Directory holding BraTS case folders.
        checkpoint_dir: Directory containing local model checkpoints.
        registry_dir: Root of the versioned model registry.
        faster_rcnn_registry_version: Registry version served for Faster R-CNN.
        yolo_registry_version: Registry version served for YOLOv8.
        swin_unetr_weights: SwinUNETR BRATS21 checkpoint for 3D segmentation.
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    database_url: str = "sqlite:///./brain_tumor.db"
    upload_dir: Path = _PROJECT_ROOT / "uploads"
    max_upload_size_mb: int = 10
    frontend_url: str = "http://localhost:8001"
    case_viewer_dir: Path = _PROJECT_ROOT / "data" / "BraTS2021_Training_Data"

    checkpoint_dir: Path = _PROJECT_ROOT / "checkpoints"
    registry_dir: Path = _PROJECT_ROOT / "registry"
    faster_rcnn_registry_version: str = "v1.0.0"
    yolo_registry_version: str = "v1.0.0"
    swin_unetr_weights: Path = (
        _PROJECT_ROOT / ".weights" / "fold0_f48_ep300_4gpu_dice0_8854" / "model.pt"
    )

    @field_validator("secret_key")
    @classmethod
    def _require_secret_key(cls, value: str) -> str:
        """Reject missing, placeholder, or short JWT signing keys."""
        if value in _PLACEHOLDER_SECRETS or len(value) < 32:
            raise ValueError(
                "SECRET_KEY must be set to a random value of at least 32 characters"
            )
        return value

    @field_validator("database_url")
    @classmethod
    def _anchor_sqlite_path(cls, value: str) -> str:
        """Resolve relative SQLite URLs against the backend directory."""
        return resolve_database_url(value)

    @field_validator("upload_dir")
    @classmethod
    def _anchor_upload_dir(cls, value: Path) -> Path:
        """Resolve a relative upload dir against the project root, not the cwd."""
        return value if value.is_absolute() else (_PROJECT_ROOT / value).resolve()

    @field_validator("faster_rcnn_registry_version", "yolo_registry_version")
    @classmethod
    def _registry_version_contained(cls, value: str) -> str:
        """Reject registry versions that cannot be a single path segment."""
        if not _REGISTRY_VERSION_RE.fullmatch(value):
            raise ValueError("registry version must match vMAJOR.MINOR[.PATCH][-build]")
        return value

    @property
    def cors_origins(self) -> list[str]:
        """Return allowed CORS origins derived from the frontend URL."""
        return [self.frontend_url, self.frontend_url.replace("localhost", "127.0.0.1")]

    @property
    def faster_rcnn_checkpoint(self) -> Path:
        """Return the local Faster R-CNN legacy ``.pth`` checkpoint path."""
        return self.checkpoint_dir / "model.pth"

    @property
    def faster_rcnn_safetensors(self) -> Path:
        """Return preferred Faster R-CNN ``.safetensors`` path (registry then local)."""
        registry_path = (
            self.registry_dir
            / "faster_rcnn"
            / self.faster_rcnn_registry_version
            / "model.safetensors"
        )
        if registry_path.exists():
            return registry_path
        return self.checkpoint_dir / "model.safetensors"

    @property
    def yolo_checkpoint(self) -> Path:
        """Return the local YOLOv8 checkpoint path (registry preferred)."""
        registry_path = (
            self.registry_dir / "yolo" / self.yolo_registry_version / "yolov8_tumor.pt"
        )
        if registry_path.exists():
            return registry_path
        return self.checkpoint_dir / "yolov8_tumor.pt"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
