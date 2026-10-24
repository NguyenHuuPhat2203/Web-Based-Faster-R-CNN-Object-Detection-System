"""Pytest fixtures and configuration for backend tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production-012345")
# Integration tests persist uploads; keep them out of the real uploads/ dir.
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="brain-tumor-test-uploads-")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.core.database as database_module

TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TEST_SESSION = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)
database_module.engine = TEST_ENGINE
database_module.SessionLocal = TEST_SESSION

from backend.core.database import Base, get_db  # noqa: E402
from backend.core.rate_limit import reset_auth_rate_limit  # noqa: E402
from backend.main import app  # noqa: E402


def override_get_db() -> Generator:
    """Yield an in-memory database session for tests."""
    db = TEST_SESSION()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _auth_rate_limit_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Keep suite tests off the production auth rate-limit ceiling."""
    monkeypatch.setattr("backend.core.rate_limit.MAX_HITS", 10_000)
    reset_auth_rate_limit()
    yield
    reset_auth_rate_limit()


@pytest.fixture
def db_session() -> Generator:
    """Provide a database session with a fresh schema for unit tests."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    db = TEST_SESSION()
    try:
        yield db
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI test client backed by an in-memory database."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Register a test user and return Bearer auth headers."""
    client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "username": "TestUser",
            "password": "test123456",
        },
    )
    return _login(client, "test@example.com", "test123456")


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    """Log in and return authorization headers."""
    response = client.post("/auth/login", json={"email": email, "password": password})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_image_bytes() -> bytes:
    """Return a minimal valid PNG image (64x64, RGB)."""
    import struct
    import zlib

    def _make_png(width: int = 64, height: int = 64) -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        raw = b""
        for _ in range(height):
            raw += b"\x00" + b"\x80\x80\x80" * width
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat = (
            struct.pack(">I", len(compressed))
            + b"IDAT"
            + compressed
            + struct.pack(">I", idat_crc)
        )
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return signature + ihdr + idat + iend

    return _make_png()
