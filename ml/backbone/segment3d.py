"""3D brain-tumor segmentation (SwinUNETR BRATS21 pretrained).

Matches the official MONAI SwinUNETR BRATS21 test pipeline exactly:
4-channel (flair, t1, t1ce, t2) -> channel-wise z-score over nonzero voxels
(NormalizeIntensityd nonzero+channel_wise) -> 128^3 sliding window
(sigmoid > 0.5) -> labels 1=NCR, 2=ED, 4=ET in the native grid.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

from ml.core.labels import BY_CHANNEL, BY_LABEL, REGIONS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = (
    _REPO_ROOT / ".weights" / "fold0_f48_ep300_4gpu_dice0_8854" / "model.pt"
)

_MODALITIES = ("flair", "t1", "t1ce", "t2")

# ponytail: naive 128^3 sliding window, sw_batch 1 -> ~55s on RTX 3050 Ti 4GB,
# ~6min CPU. Faster paths (bigger sw_batch / overlap 0.5) later if needed.
_INFER_OVERLAP = 0.6
_ROI = (128, 128, 128)


class SwinUNETRSegmentor:
    """Lazy singleton around the SwinUNETR BRATS21 fold-0 checkpoint."""

    _instance: SwinUNETRSegmentor | None = None

    def __init__(self, weights_path: str | Path | None = None) -> None:
        """Build the segmentor (weights path, device, model)."""
        self.weights_path = Path(
            weights_path
            or os.environ.get("NEUROSCAN_SWINUNETR_WEIGHTS")
            or _DEFAULT_WEIGHTS
        )
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"SwinUNETR weights not found at {self.weights_path}. "
                "Download fold0 zip from the MONAI research-contributions "
                "release and extract model.pt."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    @classmethod
    def get(cls) -> SwinUNETRSegmentor:
        """Return the shared lazy singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_model(self) -> torch.nn.Module:
        """Load the BRATS21 fold-0 checkpoint into a SwinUNETR."""
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )
        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        return model

    def segment_volume(
        self,
        case_dir: str | Path,
        *,
        overlap: float = _INFER_OVERLAP,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Segment a BraTS case dir; returns (mask 3D uint8, affine, seconds)."""
        case_dir = Path(case_dir)
        images: list[np.ndarray] = []
        affine: np.ndarray | None = None
        for mod in _MODALITIES:
            match = sorted(case_dir.glob(f"*_{mod}.nii*"))
            if not match:
                raise FileNotFoundError(f"missing {mod} modality in {case_dir}")
            img = nib.load(match[0])
            images.append(img.get_fdata(dtype=np.float32))
            affine = img.affine
        volume = np.stack(images, axis=0)  # (4, D, H, W)
        assert affine is not None

        # NormalizeIntensityd(nonzero=True, channel_wise=True) — z-score per channel
        for c in range(volume.shape[0]):
            vals = volume[c][volume[c] > 0]
            if vals.size:
                mean, std = float(vals.mean()), float(vals.std())
                if std > 0:
                    volume[c] = (volume[c] - mean) / std

        tensor = torch.from_numpy(volume).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else torch.inference_mode()
        )
        with torch.no_grad(), ctx:
            prob = torch.sigmoid(
                sliding_window_inference(
                    tensor,
                    roi_size=_ROI,
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=overlap,
                )
            )
        seconds = time.perf_counter() - start

        prob_np = prob[0].detach().cpu().numpy()  # (3, D, H, W)
        binary = prob_np > 0.5
        mask = np.zeros(binary.shape[1:], dtype=np.uint8)
        for channel, region in BY_CHANNEL.items():
            mask[binary[channel]] = region.label
        return mask, affine, seconds

    def volumes_cm3(self, mask: np.ndarray, affine: np.ndarray) -> dict[str, float]:
        """Region volumes in cm^3 from the affine voxel spacing."""
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        voxel_cm3 = float(np.prod(spacing) / 1000.0)
        return {
            region.name: float(np.sum(mask == region.label)) * voxel_cm3
            for region in REGIONS
        }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen-Dice coefficient between two binary masks."""
    inter = float(np.sum((a > 0) & (b > 0)))
    denom = float(np.sum(a > 0) + np.sum(b > 0))
    return 2 * inter / denom if denom else 1.0


if __name__ == "__main__":
    # spike: python ml/backbone/segment3d.py <BraTS case dir>
    import sys

    case = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/BraTS2021_Training_Data/BraTS2021_01045"
    )
    seg = SwinUNETRSegmentor.get()
    print(f"device: {seg.device}  weights: {seg.weights_path.name}")
    t0 = time.perf_counter()
    seg, affine, infer_s = SwinUNETRSegmentor.get().segment_volume(case)
    print(
        f"inference: {infer_s:.1f}s  total: {time.perf_counter() - t0:.1f}s "
        f"shape: {seg.shape}"
    )
    vols = SwinUNETRSegmentor.get().volumes_cm3(seg, affine)
    print("volumes cm3:", {k: round(v, 1) for k, v in vols.items()})
    assert len(vols) == 3 and all(v >= 0 for v in vols.values())
    assert seg.dtype == np.uint8 and set(np.unique(seg)) <= {0, 1, 2, 4}
    out = case.parent / f"{case.name}_pred3d.nii.gz"
    nib.save(nib.Nifti1Image(seg, affine), out)
    print(f"saved: {out}")
    expert = sorted(case.glob("*_seg.nii*"))
    if expert:
        gt = nib.load(expert[0]).get_fdata().astype(np.uint8)
        print(
            "dice vs expert:",
            {
                region.name: round(dice(seg == region.label, gt == region.label), 3)
                for region in REGIONS
            },
            "| whole:",
            round(dice(seg, gt), 3),
        )
