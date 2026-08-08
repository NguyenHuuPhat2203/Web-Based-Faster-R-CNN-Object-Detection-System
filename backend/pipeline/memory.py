"""GPU memory helpers for inference pipelines."""

from __future__ import annotations

import gc

import torch


def free_gpu_memory() -> None:
    """Release reclaimable GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
