"""Image transforms for inference and shared preprocessing.

Medical MRI: no color jitter. Train augmentations live in training/data.
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import v2


def pil_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB."""
    return image.convert("RGB")


def create_inference_transform() -> v2.Compose:
    """Build inference-only transform: ToTensor in ``[0, 1]``, no augment.

    Returns:
        torchvision v2 compose suitable for Faster R-CNN inputs.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def image_to_tensor(
    image: Image.Image, device: torch.device | None = None
) -> torch.Tensor:
    """Convert a PIL RGB image to a CHW float tensor.

    Args:
        image: PIL image (will be converted to RGB).
        device: Optional target device.

    Returns:
        Tensor with shape ``[3, H, W]`` in range ``[0, 1]``.
    """
    rgb = pil_to_rgb(image)
    tensor = create_inference_transform()(rgb)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 3, (
        f"Expected 3-channel RGB, got shape {tuple(tensor.shape)}"
    )
    if device is not None:
        tensor = tensor.to(device)
    return tensor
