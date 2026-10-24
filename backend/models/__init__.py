"""SQLAlchemy ORM models for users, images, detections, and token blacklist."""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

from ..core.database import Base
from ..core.security import utcnow


class UTCDateTime(TypeDecorator):
    """Store naive UTC in the database and always return timezone-aware UTC.

    SQLite drops tzinfo, so a plain ``DateTime`` returns naive values that
    serialize without an offset and cannot be compared with aware datetimes.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Convert to naive UTC; naive inputs are assumed to already be UTC."""
        if value is None or value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Attach UTC to the naive value read from the database."""
        return None if value is None else value.replace(tzinfo=UTC)


class User(Base):
    """Registered application user."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True)
    created_at = Column(UTCDateTime, default=utcnow)

    images = relationship("Image", back_populates="user", cascade="all, delete-orphan")


class TokenBlacklist(Base):
    """Revoked JWT records checked on protected requests."""

    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(UTCDateTime, nullable=False)
    created_at = Column(UTCDateTime, default=utcnow)


class Image(Base):
    """Uploaded scan image and associated inference metadata."""

    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    original_name = Column(String(500), nullable=False)
    stored_name = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    mime_type = Column(String(50), nullable=False)
    uploaded_at = Column(UTCDateTime, default=utcnow)
    deleted = Column(Boolean, default=False)

    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=True)

    model_type = Column(String(50), nullable=True)
    model_version = Column(String(100), nullable=True)
    threshold = Column(Float, nullable=True)

    user = relationship("User", back_populates="images")
    detections = relationship(
        "Detection",
        back_populates="image",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Detection(Base):
    """Single object-detection result for an uploaded image."""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    label = Column(Integer, nullable=False)
    label_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)

    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    image = relationship("Image", back_populates="detections")


# Composite indexes mirrored from migration 003 so autogenerate sees no drift.
Index(
    "ix_images_user_id_deleted_uploaded_at",
    Image.user_id,
    Image.deleted,
    Image.uploaded_at.desc(),
)
Index("ix_images_user_id_deleted", Image.user_id, Image.deleted)
Index("ix_detections_image_id_score", Detection.image_id, Detection.score.desc())
