"""Regression: relative SQLite URLs must not follow process cwd."""

from __future__ import annotations

from pathlib import Path

from backend.config import _BACKEND_DIR, resolve_database_url


def test_relative_sqlite_url_anchors_to_backend_dir() -> None:
    """Repo-root cwd must still open backend/brain_tumor.db, not ./brain_tumor.db."""
    resolved = resolve_database_url("sqlite:///./brain_tumor.db")
    assert resolved == (
        f"sqlite:///{(_BACKEND_DIR / 'brain_tumor.db').resolve().as_posix()}"
    )


def test_absolute_sqlite_url_unchanged() -> None:
    """Absolute SQLite paths are left alone."""
    absolute = Path("D:/data/app.db")
    url = f"sqlite:///{absolute.as_posix()}"
    assert resolve_database_url(url) == url


def test_non_sqlite_url_unchanged() -> None:
    """Postgres (etc.) URLs are not rewritten."""
    url = "postgresql+psycopg://user:pass@localhost/db"
    assert resolve_database_url(url) == url
