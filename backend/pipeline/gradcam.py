"""Grad-CAM heatmap generation for Faster R-CNN (serve pipeline only)."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image as PILImage
from torch import nn


class GradCAM:
    """Grad-CAM over the last ResNet-50 bottleneck of a torchvision Faster R-CNN."""

    def __init__(self, model: nn.Module) -> None:
        """Bind to the backbone's final conv layer (hooks are per-call)."""
        self.model = model
        self.target_layer = model.backbone.body.layer4[-1].conv3  # type: ignore[union-attr,index]

    def compute(self, image: torch.Tensor, target_label: int) -> str | None:
        """Return a base64 PNG heatmap for the best detection of ``target_label``.

        Gradients are taken w.r.t. the hooked activation only, so model
        parameters never accumulate ``.grad`` and no hook outlives the call.

        Args:
            image: Image tensor ``[3, H, W]`` in ``[0, 1]`` on the model device.
            target_label: Class index of the detection to explain.

        Returns:
            Heatmap PNG (``H x W``, jet colormap) as base64, or ``None`` when the
            label is not detected or the map is flat.
        """
        activations: list[torch.Tensor] = []
        handle = self.target_layer.register_forward_hook(
            lambda _module, _inputs, output: activations.append(output)
        )
        try:
            with torch.enable_grad():
                prediction = self.model([image])[0]
                matches = (prediction["labels"] == target_label).nonzero().flatten()
                if matches.numel() == 0 or not activations:
                    return None
                best = matches[prediction["scores"][matches].argmax()]
                activation = activations[0]
                (gradient,) = torch.autograd.grad(
                    prediction["scores"][best], activation
                )
        finally:
            handle.remove()

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True)).detach()

        # The backbone saw the resized, bottom/right zero-padded batch tensor:
        # map the CAM onto that canvas, crop the padding, then resize to input.
        transformed, _ = self.model.transform([image])  # type: ignore[operator]
        padded_height, padded_width = transformed.tensors.shape[-2:]
        resized_height, resized_width = transformed.image_sizes[0]
        cam = F.interpolate(
            cam,
            size=(padded_height, padded_width),
            mode="bilinear",
            align_corners=False,
        )[..., :resized_height, :resized_width]
        _, height, width = image.shape
        cam = F.interpolate(
            cam, size=(height, width), mode="bilinear", align_corners=False
        )
        heatmap = cam[0, 0].cpu().numpy()

        low, high = float(heatmap.min()), float(heatmap.max())
        if high - low < 1e-8:
            return None
        heatmap = (heatmap - low) / (high - low)

        rgb = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
        buffer = io.BytesIO()
        PILImage.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
