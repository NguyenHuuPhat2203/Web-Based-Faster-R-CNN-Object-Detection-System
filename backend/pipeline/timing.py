"""Per-stage timing helper for inference pipelines."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

from ..core.metrics import INFERENCE_STAGE_LATENCY


@contextmanager
def stage_timer(stage: str, *, model_version: str) -> Iterator[None]:
    """Time one pipeline stage.

    Binds ``{stage}_ms`` to the structlog context (merged into the pipeline's
    own ``inference_stage_timing`` log event) and records the duration in the
    ``inference_stage_latency_ms`` histogram.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        structlog.contextvars.bind_contextvars(**{f"{stage}_ms": round(elapsed_ms, 3)})
        INFERENCE_STAGE_LATENCY.labels(
            stage=stage, model_version=model_version
        ).observe(elapsed_ms)
