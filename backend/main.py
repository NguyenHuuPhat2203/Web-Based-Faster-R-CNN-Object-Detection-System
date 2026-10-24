"""Brain Tumor Detector FastAPI application entry point."""

from __future__ import annotations

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from ml.core.constants import FASTER_RCNN_MODEL_VERSION

from .api.routers import auth, case_viewer, images, predict
from .config import get_settings
from .core.database import SessionLocal
from .core.exceptions import (
    ConflictError,
    DomainError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    UnauthorizedError,
    ValidationError,
)
from .core.lifespan import lifespan
from .core.logging import configure_structlog
from .core.metrics import metrics_payload
from .core.observability import ObservabilityMiddleware

# Most specific class wins (Starlette walks the exception MRO).
_DOMAIN_ERROR_STATUS: dict[type[DomainError], int] = {
    DomainError: 400,
    ValidationError: 400,
    PayloadTooLargeError: 413,
    UnauthorizedError: 401,
    ForbiddenError: 403,
    NotFoundError: 404,
    ConflictError: 409,
}

configure_structlog(json_logs=True)
settings = get_settings()

app = FastAPI(title="Brain Tumor Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ObservabilityMiddleware)


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a domain error to its HTTP status with a ``detail`` body."""
    assert isinstance(exc, DomainError)
    status = next(
        _DOMAIN_ERROR_STATUS[cls]
        for cls in type(exc).__mro__
        if cls in _DOMAIN_ERROR_STATUS
    )
    return JSONResponse(status_code=status, content={"detail": exc.message})


for _error_class in _DOMAIN_ERROR_STATUS:
    app.add_exception_handler(_error_class, _domain_error_handler)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(images.router)
app.include_router(case_viewer.router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return service status and runtime device information."""
    return {
        "status": "Backend is running",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/healthz")
def healthz() -> dict[str, str | bool | int]:
    """Return liveness probe with model version metadata."""
    return {
        "status": "ok",
        "model_version": FASTER_RCNN_MODEL_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }


@app.get("/readyz", response_model=None)
def readyz() -> dict[str, str] | JSONResponse:
    """Return readiness probe: database must accept a trivial query."""
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")
