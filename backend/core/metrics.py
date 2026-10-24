"""Prometheus metrics for the inference service."""

from prometheus_client import Counter, Histogram, generate_latest

INFERENCE_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000],
    labelnames=["model_version", "endpoint"],
)

INFERENCE_THROUGHPUT = Counter(
    "inference_throughput_total",
    "Total inference requests processed",
    labelnames=["model_version"],
)

INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Total inference errors",
    labelnames=["error_type", "endpoint"],
)

INFERENCE_STAGE_LATENCY = Histogram(
    "inference_stage_latency_ms",
    "Per-stage inference latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500],
    labelnames=["stage", "model_version"],
)


def metrics_payload() -> bytes:
    """Return Prometheus text exposition payload."""
    return generate_latest()
