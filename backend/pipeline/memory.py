"""GPU memory helpers for inference pipelines."""

from __future__ import annotations

import gc

import torch


def free_gpu_memory() -> None:
    """Release reclaimable GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    """Return True when an exception indicates CUDA out-of-memory.

    Catches both the typed ``torch.cuda.OutOfMemoryError`` and the
    message-based variant some wrappers (e.g. ultralytics) raise.
    """
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
    )
