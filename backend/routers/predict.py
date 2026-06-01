"""Protected prediction route — Faster R-CNN with optional Grad-CAM heatmap."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import uuid

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from PIL import Image
from sqlalchemy.orm import Session

from database import get_db
from models import Detection, Image as ImageModel, User
from schemas import PredictionResult
from dependencies import get_current_user
import config as cfg

router = APIRouter(tags=["detection"])

# ── Model initialisation (lazy, on first request) ────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# Class ID → human-readable name (1 = background in TorchVision convention)
CLASS_NAMES = {1: "Brain tumor", 2: "Brain tumor"}
_model = None

# ── Grad-CAM helpers ─────────────────────────────────────────────────


class GradCAM:
    """Compute a Grad-CAM heatmap for a Faster R-CNN prediction.

    Hooks into the last convolutional layer of the ResNet-50 backbone
    (layer4[2].conv3) to capture activations and gradients.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Hook the last conv layer of the backbone
        self.target_layer = model.backbone.body.layer4[2].conv3
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def compute(self, image_tensor: torch.Tensor, target_label: int) -> str | None:
        """Run a forward + backward pass and return the heatmap as a base64 PNG.

        Args:
            image_tensor: [1, 3, H, W] normalized image on the correct device.
            target_label: The class index to compute the gradient for.

        Returns:
            Base64-encoded PNG heatmap, or None on failure.
        """
        self.model.zero_grad()

        # Forward
        with torch.set_grad_enabled(True):
            predictions = self.model(image_tensor)

        if len(predictions) == 0:
            return None

        pred = predictions[0]
        if len(pred["scores"]) == 0:
            return None

        # Find the highest-scoring detection of the target class
        best_idx = None
        best_score = -1
        for i, (label, score) in enumerate(zip(pred["labels"], pred["scores"])):
            if label == target_label and score > best_score:
                best_idx = i
                best_score = score

        if best_idx is None:
            return None

        # Backward from the class logit
        score = pred["scores"][best_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            return None

        # Grad-CAM weighting
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, Hf, Wf]
        cam = F.relu(cam)

        # Resize to original image size
        _, _, H, W = image_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return None
        cam = (cam - cam_min) / (cam_max - cam_min)

        # Apply jet colormap and encode as PNG
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(cam, cmap="jet", alpha=1.0, vmin=0, vmax=1)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_model() -> tuple[torch.nn.Module, GradCAM | None]:
    global _model
    if _model is not None:
        # _model is stored as (model, gradcam)
        return _model

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Try local checkpoint first, fall back to Hugging Face Hub
    checkpoint_path = "../checkpoints/model.pth"
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"[_load_model] Loaded checkpoint from {checkpoint_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
            print("[_load_model] Local checkpoint not found, downloading from HF Hub...")
            hf_path = hf_hub_download(
                repo_id=cfg.HF_CHECKPOINT_REPO,
                filename=cfg.HF_CHECKPOINT_FILENAME,
                token=cfg.HF_ACCESS_TOKEN,
            )
            print(f"[_load_model] Downloaded to {hf_path}")
            state = torch.load(hf_path, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            print("[_load_model] Checkpoint loaded from HF Hub successfully")
        except Exception as e:
            print(f"[_load_model] HF Hub download FAILED: {e}")
            print("[_load_model] Running with COCO backbone + RANDOM head!")

    model.to(DEVICE)
    model.eval()
    gradcam = GradCAM(model)
    _model = (model, gradcam)
    return _model


@router.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ── Validate ──────────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    contents = await file.read()
    if len(contents) > cfg.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {cfg.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    # ── Inference ─────────────────────────────────────────────────────
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = torchvision.transforms.ToTensor()(image).to(DEVICE)

    model, gradcam = _load_model()
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    mask = pred["scores"] > threshold

    boxes = pred["boxes"][mask].cpu().numpy().tolist()
    labels = pred["labels"][mask].cpu().numpy().tolist()
    scores = pred["scores"][mask].cpu().numpy().tolist()
    label_names = [CLASS_NAMES.get(label, f"Class {label}") for label in labels]

    # ── Grad-CAM heatmap (for best detection) ─────────────────────────
    heatmap: str | None = None
    if len(labels) > 0 and gradcam is not None:
        try:
            heatmap = gradcam.compute(img_tensor.unsqueeze(0), labels[0])
        except Exception:
            heatmap = None  # Graceful fallback

    # ── Save to disk + DB ──
    user_dir = os.path.join(cfg.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, stored_name)

    image.save(file_path)

    db_image = ImageModel(
        user_id=current_user.id,
        original_name=file.filename or "unknown",
        stored_name=stored_name,
        filepath=file_path,
        mime_type=file.content_type or "image/jpeg",
        width=image.width,
        height=image.height,
        file_size=len(contents),
        content_hash=hashlib.sha256(contents).hexdigest(),
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
        threshold=threshold,
    )
    db.add(db_image)
    db.flush()

    for box, label, score, label_name in zip(boxes, labels, scores, label_names):
        db.add(Detection(
            image_id=db_image.id,
            label=label,
            label_name=label_name,
            score=score,
            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        ))
    db.commit()
    db.refresh(db_image)

    return PredictionResult(
        boxes=boxes, labels=labels, scores=scores, label_names=label_names,
        heatmap=heatmap,
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
    )
