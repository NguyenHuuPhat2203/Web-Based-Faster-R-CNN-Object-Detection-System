"""HTTP middleware for correlation IDs and inference timing."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ml.core.constants import FASTER_RCNN_MODEL_VERSION, YOLO_MODEL_VERSION

from .metrics import INFERENCE_ERRORS, INFERENCE_LATENCY, INFERENCE_THROUGHPUT

logger = structlog.get_logger()

# Exact paths only: arbitrary request paths as metric labels are unbounded.
_MODEL_VERSION_BY_PATH = {
    "/predict": FASTER_RCNN_MODEL_VERSION,
    "/v2/predict": FASTER_RCNN_MODEL_VERSION,
    "/predict-yolo": YOLO_MODEL_VERSION,
    "/v2/predict/yolo": YOLO_MODEL_VERSION,
}
_CORRELATION_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _correlation_id(request: Request) -> str:
    """Return the caller's correlation ID if well-formed, else a fresh UUID."""
    supplied = request.headers.get("x-correlation-id", "")
    if _CORRELATION_ID_PATTERN.fullmatch(supplied):
        return supplied
    return str(uuid.uuid4())


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach correlation_id and record latency for predict endpoints."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request with structured logging context."""
        correlation_id = _correlation_id(request)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        started = time.perf_counter()
        error_type: str | None = None
        response: Response | None = None
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                error_type = "http_error"
            response.headers["x-correlation-id"] = correlation_id
            return response
        except Exception:
            error_type = "unhandled"
            raise
        finally:
            path = request.url.path
            model_version = _MODEL_VERSION_BY_PATH.get(path)
            if model_version is not None:
                latency_ms = (time.perf_counter() - started) * 1000
                INFERENCE_LATENCY.labels(
                    model_version=model_version, endpoint=path
                ).observe(latency_ms)
                INFERENCE_THROUGHPUT.labels(model_version=model_version).inc()
                if error_type is not None:
                    INFERENCE_ERRORS.labels(error_type=error_type, endpoint=path).inc()
                logger.info(
                    "inference_request",
                    endpoint=path,
                    model_version=model_version,
                    latency_ms=round(latency_ms, 2),
                    status="success" if error_type is None else "error",
                )
