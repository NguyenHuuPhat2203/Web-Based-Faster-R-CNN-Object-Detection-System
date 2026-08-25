"""Brain Tumor Detector FastAPI application entry point."""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ml.core.constants import FRCNN_MODEL_VERSION

from .api.routers import auth, case_viewer, images, predict, predict_v2, predict_yolo
from .config import get_settings
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


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """Map not-found domain errors to HTTP 404 responses."""
    return JSONResponse(status_code=404, content={"detail": exc.message})


@app.exception_handler(ForbiddenError)
async def forbidden_handler(request: Request, exc: ForbiddenError) -> JSONResponse:
    """Map forbidden domain errors to HTTP 403 responses."""
    return JSONResponse(status_code=403, content={"detail": exc.message})


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    """Map conflict domain errors to HTTP 409 responses."""
    return JSONResponse(status_code=409, content={"detail": exc.message})


@app.exception_handler(PayloadTooLargeError)
async def payload_too_large_handler(
    request: Request,
    exc: PayloadTooLargeError,
) -> JSONResponse:
    """Map payload-too-large errors to HTTP 413 responses."""
    return JSONResponse(status_code=413, content={"detail": exc.message})


@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Map validation domain errors to HTTP 400 responses."""
    return JSONResponse(status_code=400, content={"detail": exc.message})


@app.exception_handler(UnauthorizedError)
async def unauthorized_handler(
    request: Request,
    exc: UnauthorizedError,
) -> JSONResponse:
    """Map unauthorized domain errors to HTTP 401 responses."""
    return JSONResponse(status_code=401, content={"detail": exc.message})


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Map remaining domain errors to HTTP 400 responses."""
    return JSONResponse(status_code=400, content={"detail": exc.message})


app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(predict_yolo.router)
app.include_router(predict_v2.router)
app.include_router(images.router)
app.include_router(case_viewer.router)

if os.path.exists(settings.upload_dir):
    app.mount(
        "/static/uploads",
        StaticFiles(directory=settings.upload_dir),
        name="uploads",
    )


@app.get("/")
def read_root() -> dict[str, str]:
    """Return service status and runtime device information."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "Backend is running",
        "device": device,
        "auth": "enabled",
    }


@app.get("/healthz")
def healthz() -> dict[str, str | bool | int]:
    """Return liveness probe with model version metadata."""
    return {
        "status": "ok",
        "model_version": FRCNN_MODEL_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.get("/readyz", response_model=None)
def readyz() -> dict[str, str] | JSONResponse:
    """Return readiness probe: database must accept a trivial query."""
    try:
        from sqlalchemy import text

        from .core.database import SessionLocal

        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable"},
        )
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")
