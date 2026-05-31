"""Pytest fixtures and configuration for backend tests."""

from __future__ import annotations

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# IMPORTANT: Patch database engine BEFORE any app code imports it.
# We use StaticPool so all connections share the same in-memory SQLite DB.
import database
TEST_ENGINE = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
TEST_SESSION = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)
database.engine = TEST_ENGINE
database.SessionLocal = TEST_SESSION

# Now it's safe to import app code — it will use our patched engine
import models  # noqa: F401, E402 — register tables with Base.metadata
from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

def override_get_db() -> Generator:
    db = TEST_SESSION()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client() -> Generator:
    Base.metadata.create_all(bind=TEST_ENGINE)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Register a test user and return Bearer auth headers."""
    client.post("/auth/register", json={
        "email": "test@example.com",
        "username": "TestUser",
        "password": "test123456",
    })
    return _login(client, "test@example.com", "test123456")


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    resp = client.post("/auth/login", json={"email": email, "password": password})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_image_bytes() -> bytes:
    """Return a minimal valid PNG image (64×64, RGB)."""
    import struct
    import zlib

    def _make_png(width: int = 64, height: int = 64) -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        raw = b""
        for _ in range(height):
            raw += b"\x00" + b"\x80\x80\x80" * width
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend

    return _make_png()
