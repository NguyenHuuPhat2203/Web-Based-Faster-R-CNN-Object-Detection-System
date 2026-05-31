from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship

from database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=True)   # null for Google‑only users
    google_id = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    images = relationship("Image", back_populates="user")


class TokenBlacklist(Base):
    """Revoked JWTs – checked on every protected request."""
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)           # so we can purge stale rows
    created_at = Column(DateTime, default=_utcnow)


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_name = Column(String(500), nullable=False)
    stored_name = Column(String(255), nullable=False)       # uuid‑based file on disk
    filepath = Column(String(1000), nullable=False)
    mime_type = Column(String(50), nullable=False)
    detection_result = Column(JSON, nullable=True)          # { boxes, labels, scores }
    uploaded_at = Column(DateTime, default=_utcnow)
    deleted = Column(Boolean, default=False)                # soft delete

    user = relationship("User", back_populates="images")
