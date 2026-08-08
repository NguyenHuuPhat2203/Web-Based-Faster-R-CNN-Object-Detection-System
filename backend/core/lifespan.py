"""Application startup and shutdown hooks."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import get_settings
from ..repositories.token_repository import TokenRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup tasks and ensure upload directories exist.

    Args:
        app: FastAPI application instance.

    Yields:
        Control back to the application runtime.
    """
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    from .database import SessionLocal

    with SessionLocal() as db:
        TokenRepository(db).purge_expired()

    yield
