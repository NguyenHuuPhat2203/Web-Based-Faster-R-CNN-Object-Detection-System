"""In-process sliding-window rate limiter for auth endpoints."""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60.0
MAX_HITS = 10
# Above this many tracked clients, idle buckets are swept to bound memory.
SWEEP_THRESHOLD = 10_000

# ponytail: process-global lock and per-process state; use a shared store
# (Redis) if the API runs with multiple workers.
_lock = threading.Lock()
_hits: dict[str, deque[float]] = {}


def reset_auth_rate_limit() -> None:
    """Clear recorded hits. For tests."""
    with _lock:
        _hits.clear()


def _sweep(cutoff: float) -> None:
    """Drop buckets whose newest hit is outside the window. Caller holds the lock."""
    for key in [k for k, bucket in _hits.items() if bucket[-1] <= cutoff]:
        del _hits[key]


def enforce_auth_rate_limit(request: Request) -> None:
    """Reject the request with 429 when the client exceeds MAX_HITS per window.

    The client key is ``request.client.host``. Behind a reverse proxy, run uvicorn
    with ``--proxy-headers`` and ``--forwarded-allow-ips`` so this is the real
    client address; ``X-Forwarded-For`` is never read here because clients can
    spoof it.
    """
    key = request.client.host if request.client is not None else "unknown"
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS
    with _lock:
        if len(_hits) > SWEEP_THRESHOLD:
            _sweep(cutoff)
        bucket = _hits.setdefault(key, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= MAX_HITS:
            retry_after = math.ceil(bucket[0] + WINDOW_SECONDS - now)
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(max(retry_after, 1))},
            )
        bucket.append(now)
