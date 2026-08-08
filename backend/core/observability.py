"""HTTP middleware for correlation IDs and inference timing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ml.core.constants import FRCNN_MODEL_VERSION, YOLO_MODEL_VERSION

from .metrics import INFERENCE_ERRORS, INFERENCE_LATENCY, INFERENCE_THROUGHPUT

logger = structlog.get_logger()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach correlation_id and record latency for predict endpoints."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with structured logging context."""
        correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        started = time.perf_counter()
        status = "success"
        response: Response | None = None
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                status = "error"
            return response
        except Exception:
            status = "error"
            INFERENCE_ERRORS.labels(
                error_type="unhandled", endpoint=request.url.path
            ).inc()
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            path = request.url.path
            if path.startswith("/predict") or path.startswith("/v2/predict"):
                model_version = (
                    YOLO_MODEL_VERSION if "yolo" in path else FRCNN_MODEL_VERSION
                )
                INFERENCE_LATENCY.labels(
                    model_version=model_version, endpoint=path
                ).observe(latency_ms)
                INFERENCE_THROUGHPUT.labels(model_version=model_version).inc()
                if status == "error":
                    INFERENCE_ERRORS.labels(
                        error_type="http_error", endpoint=path
                    ).inc()
                logger.info(
                    "inference_request",
                    endpoint=path,
                    model_version=model_version,
                    latency_ms=round(latency_ms, 2),
                    status=status,
                    input_shape=None,
                )
            if response is not None:
                response.headers["x-correlation-id"] = correlation_id
