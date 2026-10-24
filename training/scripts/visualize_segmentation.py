#!/usr/bin/env python3
"""Visualize a MedNeXt segmentation checkpoint on one BraTS case.

Produces <case>_pred.nii.gz plus a PNG overlay (axial/coronal/sagittal).
Auto-detects n_classes from the checkpoint (3 = legacy TC/WT/ET head,
4 = disjoint bg/NCR/ED/ET).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODALITIES = ("flair", "t1", "t1ce", "t2")
_REGION_COLORS = {1: "#f59e0b", 2: "#10b981", 4: "#ef4444"}
_VIEWS = ("axial", "coronal", "sagittal")


def _class_to_label(n_classes: int) -> dict[int, int]:
    if n_classes == 4:
        return {1: 1, 2: 2, 3: 4}
    return {0: 1, 1: 2, 2: 4}


_XS_CONFIG = {
    "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
    "deep_supervision": False,
    "do_res": True,
    "do_res_up_down": True,
    "block_counts": [2] * 9,
    "norm_type": "group",
    "dim": "3d",
}


def _build_model(state: dict, n_classes: int) -> torch.nn.Module:
    """Build MedNeXt matching the checkpoint (XS layout or legacy defaults)."""
    from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

    last_error: RuntimeError | None = None
    for extra in (_XS_CONFIG, {}):
        model = MedNeXt(
            in_channels=4,
            n_channels=32,
            n_classes=n_classes,
            kernel_size=5,
            grn=True,
            **extra,
        )
        try:
            model.load_state_dict(state, strict=True)
            return model
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError("checkpoint matches no known MedNeXt layout") from last_error


def predict_volume(
    weights_path: Path,
    case_dir: Path,
    *,
    overlap: float = 0.5,
    patch_size: int = 96,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mask uint8 in native grid, affine, seconds)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    n_classes = int(state["out_0.conv_out.weight"].shape[1])
    model = _build_model(state, n_classes)
    model.eval().to(device)

    images: list[np.ndarray] = []
    affine: np.ndarray | None = None
    for modality in _MODALITIES:
        img = nib.load(sorted(case_dir.glob(f"*_{modality}.nii*"))[0])
        images.append(img.get_fdata(dtype=np.float32))
        affine = img.affine
    assert affine is not None
    to_ras = ornt_transform(io_orientation(affine), axcodes2ornt("RAS"))
    back = ornt_transform(axcodes2ornt("RAS"), io_orientation(affine))
    volume = np.stack([apply_orientation(x, to_ras) for x in images], axis=0)
    for channel in range(volume.shape[0]):
        vals = volume[channel][volume[channel] > 0]
        if vals.size:
            mean, std = float(vals.mean()), float(vals.std())
            if std > 0:
                volume[channel] = (volume[channel] - mean) / std

    start = time.perf_counter()
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0).to(device)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.inference_mode()
    )
    with torch.no_grad(), ctx:
        logits = sliding_window_inference(
            tensor,
            roi_size=(patch_size,) * 3,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
        )
    seconds = time.perf_counter() - start

    cls = logits[0].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
    cls = apply_orientation(cls, back)
    mapping = _class_to_label(n_classes)
    mask = np.zeros(cls.shape, dtype=np.uint8)
    for class_idx, label_value in mapping.items():
        mask[cls == class_idx] = label_value
    return mask, affine, seconds


def save_overlay_png(
    image: np.ndarray,
    mask: np.ndarray,
    gt: np.ndarray | None,
    out_path: Path,
) -> None:
    """Three-view PNG: flair grayscale with colored prediction overlay."""
    tumor = np.argwhere(gt > 0) if gt is not None else np.argwhere(mask > 0)
    center = (
        tumor.mean(axis=0).astype(int) if len(tumor) else np.array(image.shape) // 2
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = (
        (image[center[0], :, :], mask[center[0], :, :], "axial"),
        (image[:, center[1], :], mask[:, center[1], :], "coronal"),
        (image[:, :, center[2]], mask[:, :, center[2]], "sagittal"),
    )
    for ax, (img_slice, mask_slice, name) in zip(axes, slices, strict=True):
        ax.imshow(img_slice.T, cmap="gray", origin="lower")
        rgba = np.zeros((*mask_slice.shape, 4))
        for label_value, hex_color in _REGION_COLORS.items():
            rgba[mask_slice == label_value] = matplotlib.colors.to_rgba(hex_color, 0.45)
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        default=_PROJECT_ROOT / "data" / "BraTS2021_Training_Data" / "BraTS2021_01045",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=_PROJECT_ROOT / ".weights" / "mednext_xs" / "mednext_xs_best.pth",
    )
    args = parser.parse_args(argv)

    mask, affine, seconds = predict_volume(args.weights, args.case)
    voxels = {int(v): int(np.sum(mask == v)) for v in np.unique(mask)}
    print(f"inference: {seconds:.1f}s | voxels: {voxels}")

    pred_path = args.case.parent / f"{args.case.name}_pred.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), pred_path)
    print("saved:", pred_path)

    flair = nib.load(sorted(args.case.glob("*_flair.nii*"))[0]).get_fdata()
    gt = None
    gt_files = sorted(args.case.glob("*_seg.nii*"))
    if gt_files:
        gt = nib.load(gt_files[0]).get_fdata().astype(np.uint8)
        from ml.core.labels import REGIONS
        from ml.models.swin_unetr import dice

        print(
            "dice vs expert:",
            {
                region.name: round(dice(mask == region.label, gt == region.label), 3)
                for region in REGIONS
            },
            "| whole:",
            round(dice(mask, gt), 3),
        )
    png_path = args.case.parent / f"{args.case.name}_pred_overlay.png"
    save_overlay_png(flair, mask, gt, png_path)
    print("saved:", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
