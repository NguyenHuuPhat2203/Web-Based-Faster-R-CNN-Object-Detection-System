#!/usr/bin/env python3
"""Visualize a MedNeXt segmentation checkpoint on one BraTS case.

Produces <case>_pred.nii.gz plus a PNG overlay (sagittal/coronal/axial).
The checkpoint must be the 4-class disjoint bg/NCR/ED/ET head trained in
``training/notebooks/mednext_xs_brats.ipynb``.
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

from ml.core.labels import REGIONS
from ml.models.swin_unetr import dice, normalize_nonzero

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODALITIES = ("flair", "t1", "t1ce", "t2")
# Network output class index -> BraTS label (class 0 is background).
_CLASS_TO_LABEL = {1: 1, 2: 2, 3: 4}
_REGION_COLORS = {r.label: tuple(c / 255 for c in r.color) for r in REGIONS}


def _build_model(state: dict) -> torch.nn.Module:
    """Build the MedNeXt-XS network and load ``state`` strictly."""
    from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

    model = MedNeXt(
        in_channels=4,
        n_channels=32,
        n_classes=len(_CLASS_TO_LABEL) + 1,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=5,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2] * 9,
        norm_type="group",
        dim="3d",
        grn=True,
    )
    model.load_state_dict(state, strict=True)
    return model


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
    model = _build_model(state)
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
    normalize_nonzero(volume)

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
    mask = np.zeros(cls.shape, dtype=np.uint8)
    for class_idx, label_value in _CLASS_TO_LABEL.items():
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
    # BraTS volumes are stored (x: left-right, y: front-back, z: bottom-top).
    slices = (
        (image[center[0], :, :], mask[center[0], :, :], "sagittal"),
        (image[:, center[1], :], mask[:, center[1], :], "coronal"),
        (image[:, :, center[2]], mask[:, :, center[2]], "axial"),
    )
    for ax, (img_slice, mask_slice, name) in zip(axes, slices, strict=True):
        ax.imshow(img_slice.T, cmap="gray", origin="lower")
        rgba = np.zeros((*mask_slice.shape, 4))
        for label_value, color in _REGION_COLORS.items():
            rgba[mask_slice == label_value] = (*color, 0.45)
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
