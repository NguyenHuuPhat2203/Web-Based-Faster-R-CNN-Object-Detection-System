"""Unit tests for the pipeline stage timing helper."""

from __future__ import annotations

import prometheus_client
import structlog

from backend.core.metrics import INFERENCE_STAGE_LATENCY
from backend.pipeline.timing import stage_timer


def test_stage_timer_binds_ms_to_log_context() -> None:
    """After the stage, ``{stage}_ms`` is visible in structlog context vars."""
    structlog.contextvars.clear_contextvars()
    with stage_timer("forward", model_version="resnet50-fpn-v1"):
        pass
    ctx = structlog.contextvars.get_contextvars()
    assert "forward_ms" in ctx
    assert ctx["forward_ms"] >= 0


def test_stage_timer_observes_histogram() -> None:
    """The stage duration is recorded in the Prometheus histogram."""
    structlog.contextvars.clear_contextvars()
    with stage_timer("gradcam", model_version="resnet50-fpn-v1"):
        pass
    count = prometheus_client.REGISTRY.get_sample_value(
        "inference_stage_latency_ms_count",
        {"stage": "gradcam", "model_version": "resnet50-fpn-v1"},
    )
    assert count is not None and count >= 1


def test_stage_timer_histogram_registered() -> None:
    """The stage latency metric exists in the Prometheus registry."""
    names = {m.name for m in prometheus_client.REGISTRY.collect()}
    assert "inference_stage_latency_ms" in names
    assert INFERENCE_STAGE_LATENCY._labelnames == ("stage", "model_version")
