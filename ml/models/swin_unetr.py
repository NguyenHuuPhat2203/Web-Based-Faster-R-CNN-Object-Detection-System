"""3D brain-tumor segmentation (SwinUNETR BRATS21 pretrained).

Matches the official MONAI SwinUNETR BRATS21 test pipeline:
4-channel (flair, t1ce, t1, t2) -> channel-wise z-score over nonzero voxels
(``NormalizeIntensityd(nonzero=True, channel_wise=True)``) -> 128^3 sliding
window -> sigmoid > 0.5 -> BraTS labels 1=NCR, 2=ED, 4=ET in the native grid.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

from ml.core.labels import REGIONS

# Input channel order of the BRATS21 checkpoint (brats21_folds.json "image"
# lists). Swapping t1/t1ce silently kills the enhancing-tumor channel.
MODALITIES = ("flair", "t1ce", "t1", "t2")

# Output channels are nested regions: 0=TC (tumor core), 1=WT (whole tumor),
# 2=ET (enhancing). Paint order matters: WT first, then TC, then ET on top,
# so WT-only voxels become ED, TC-only voxels NCR, ET voxels ET.
_CHANNEL_PAINT_ORDER = ((1, 2), (0, 1), (2, 4))  # (channel, BraTS label)

# ponytail: naive 128^3 sliding window, sw_batch 1 -> ~55s on RTX 3050 Ti 4GB,
# ~6min CPU. Bigger sw_batch / overlap 0.5 when latency matters.
_INFER_OVERLAP = 0.6
_ROI_SIZE = (128, 128, 128)


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score each channel over its nonzero voxels; background stays 0.

    Args:
        volume: Float array ``(C, D, H, W)``; modified in place.

    Returns:
        The same array, normalized.
    """
    for channel in volume:
        foreground = channel != 0
        values = channel[foreground]
        if values.size:
            std = float(values.std())
            channel[foreground] = (values - float(values.mean())) / (std or 1.0)
    return volume


def decode_channels(binary: np.ndarray) -> np.ndarray:
    """Convert ``(3, D, H, W)`` TC/WT/ET booleans into a BraTS label volume."""
    mask = np.zeros(binary.shape[1:], dtype=np.uint8)
    for channel, label in _CHANNEL_PAINT_ORDER:
        mask[binary[channel]] = label
    return mask


def region_volumes_cm3(mask: np.ndarray, affine: np.ndarray) -> dict[str, float]:
    """Region volumes in cm^3 from the affine voxel spacing."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_cm3 = float(np.prod(spacing) / 1000.0)
    return {
        region.name: float(np.sum(mask == region.label)) * voxel_cm3
        for region in REGIONS
    }


def dice(first: np.ndarray, second: np.ndarray) -> float:
    """Sorensen-Dice coefficient between two binary masks."""
    intersection = float(np.sum((first > 0) & (second > 0)))
    denominator = float(np.sum(first > 0) + np.sum(second > 0))
    return 2 * intersection / denominator if denominator else 1.0


class SwinUNETRSegmentor:
    """SwinUNETR BRATS21 fold-0 checkpoint wrapped for whole-case inference."""

    def __init__(self, weights_path: Path) -> None:
        """Load the checkpoint onto CUDA when available, else CPU.

        Raises:
            FileNotFoundError: If ``weights_path`` does not exist.
            RuntimeError: If the checkpoint has no ``state_dict``.
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"SwinUNETR weights not found at {self.weights_path}. "
                "Download the fold0 zip from the MONAI research-contributions "
                "release and extract model.pt."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

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
        # The MONAI release checkpoint stores ``best_acc`` as a numpy float32
        # scalar; allow exactly those globals so weights_only stays on.
        numpy_scalar = np.float32(0).__reduce__()[0]
        allowed = [
            (numpy_scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            np.dtypes.Float32DType,
            np.dtypes.Float64DType,
        ]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise RuntimeError("Invalid SwinUNETR checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(self.device)
        return model

    def segment_volume(
        self,
        case_dir: Path,
        *,
        overlap: float = _INFER_OVERLAP,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Segment a BraTS case directory.

        Returns:
            ``(mask, affine, seconds)``: uint8 label volume, NIfTI affine of the
            source grid, and model inference time.

        Raises:
            FileNotFoundError: If a modality file is missing.
        """
        images: list[np.ndarray] = []
        affine = np.eye(4)
        for modality in MODALITIES:
            matches = sorted(Path(case_dir).glob(f"*_{modality}.nii*"))
            if not matches:
                raise FileNotFoundError(f"missing {modality} modality in {case_dir}")
            image = nib.load(matches[0])
            images.append(image.get_fdata(dtype=np.float32))
            affine = image.affine
        volume = normalize_nonzero(np.stack(images, axis=0))

        tensor = torch.from_numpy(volume).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ),
        ):
            logits = sliding_window_inference(
                tensor,
                roi_size=_ROI_SIZE,
                sw_batch_size=1,
                predictor=self.model,
                overlap=overlap,
            )
        seconds = time.perf_counter() - start

        binary = torch.sigmoid(logits.float())[0].cpu().numpy() > 0.5
        return decode_channels(binary), affine, seconds


if __name__ == "__main__":
    # Manual check: python -m ml.models.swin_unetr <BraTS case dir> <model.pt>
    import sys

    case = Path(sys.argv[1])
    segmentor = SwinUNETRSegmentor(Path(sys.argv[2]))
    print(f"device: {segmentor.device}  weights: {segmentor.weights_path.name}")
    predicted, case_affine, infer_seconds = segmentor.segment_volume(case)
    print(f"inference: {infer_seconds:.1f}s  shape: {predicted.shape}")
    volumes = region_volumes_cm3(predicted, case_affine)
    print("volumes cm3:", {name: round(v, 1) for name, v in volumes.items()})
    expert = sorted(case.glob("*_seg.nii*"))
    if expert:
        truth = nib.load(expert[0]).get_fdata().astype(np.uint8)
        scores = {
            r.name: round(dice(predicted == r.label, truth == r.label), 3)
            for r in REGIONS
        }
        print("dice vs expert:", scores, "| whole:", round(dice(predicted, truth), 3))
