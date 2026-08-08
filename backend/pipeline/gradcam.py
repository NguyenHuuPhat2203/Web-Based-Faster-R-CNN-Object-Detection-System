"""Grad-CAM heatmap generation for Faster R-CNN (serve pipeline only)."""

from __future__ import annotations

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")


class GradCAM:
    """Compute a Grad-CAM heatmap for a Faster R-CNN prediction."""

    def __init__(self, model: nn.Module) -> None:
        """Register hooks on the ResNet-50 backbone's final conv layer."""
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.target_layer = model.backbone.body.layer4[2].conv3  # type: ignore[union-attr,index]
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inputs, output) -> None:
        """Store forward activations from the hooked layer."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Store backward gradients from the hooked layer."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered forward and backward hooks."""
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def compute(self, image_tensor: torch.Tensor, target_label: int) -> str | None:
        """Run forward and backward passes and return a base64 PNG heatmap.

        Args:
            image_tensor: Normalized image tensor with shape ``[1, 3, H, W]``.
            target_label: Class index used for gradient computation.

        Returns:
            Base64-encoded PNG heatmap, or ``None`` when computation fails.
        """
        self.model.zero_grad()

        with torch.set_grad_enabled(True):
            predictions = self.model(image_tensor)

        if not predictions:
            return None

        prediction = predictions[0]
        if len(prediction["scores"]) == 0:
            return None

        best_idx = None
        best_score = -1.0
        for index, (label, score) in enumerate(
            zip(prediction["labels"], prediction["scores"], strict=False)
        ):
            score_value = score.detach().item()
            if label == target_label and score_value > best_score:
                best_idx = index
                best_score = score_value

        if best_idx is None:
            return None

        score = prediction["scores"][best_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        _, _, height, width = image_tensor.shape
        cam = F.interpolate(
            cam, size=(height, width), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()  # type: ignore[assignment]

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return None
        cam = (cam - cam_min) / (cam_max - cam_min)

        figure, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        axis.imshow(cam, cmap="jet", alpha=1.0, vmin=0, vmax=1)
        axis.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(figure)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
