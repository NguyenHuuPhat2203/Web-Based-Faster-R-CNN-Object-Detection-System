"""Unit tests for the case-viewer service (BraTS NIfTI analysis)."""

from __future__ import annotations

import base64

import nibabel as nib
import numpy as np
import pytest

from backend.services.case_viewer_service import (
    analyze_case,
    list_cases,
    slice_payload,
)


def _write_case(root, name: str = "BraTS2021_00001") -> str:
    """Write a tiny synthetic BraTS-style case (t1ce + seg) and return its dir."""
    case_dir = root / name
    case_dir.mkdir(parents=True)

    seg = np.zeros((8, 8, 4), dtype=np.uint8)
    seg[2:5, 2:5, 1] = 1  # NCR: 3*3 = 9 voxels (slice 1)
    seg[1:6, 1:6, 0] = 2  # ED: 5*5 = 25 voxels (slice 0)
    seg[3:6, 3:6, 3] = 4  # ET: 3*3 = 9 voxels (slice 3)

    base = np.zeros((8, 8, 4), dtype=np.uint16)
    base[..., 1] = 200  # visible contrast on slice 1

    seg_img = nib.Nifti1Image(seg, affine=np.eye(4))
    base_img = nib.Nifti1Image(base, affine=np.eye(4))
    nib.save(seg_img, case_dir / f"{name}_seg.nii.gz")
    nib.save(base_img, case_dir / f"{name}_t1ce.nii.gz")
    return str(case_dir)


def test_analyze_case_computes_region_volumes(tmp_path) -> None:
    """Voxel counts * spacing must yield per-region cm3 volumes."""
    case_dir = _write_case(tmp_path)

    info = analyze_case(case_dir)

    assert info["case_id"] == "BraTS2021_00001"
    assert info["slices"] == 4
    # spacing 1mm -> voxel volume 1 mm3 = 0.001 cm3
    assert info["volumes_cm3"]["ncr"] == pytest.approx(9 * 0.001)
    assert info["volumes_cm3"]["ed"] == pytest.approx(25 * 0.001)
    assert info["volumes_cm3"]["et"] == pytest.approx(9 * 0.001)
    assert info["volumes_cm3"]["total"] == pytest.approx(43 * 0.001)


def test_list_cases_sorts_by_tumor_volume_desc(tmp_path) -> None:
    """Largest-tumor cases come first."""
    small = _write_case(tmp_path, "BraTS2021_00001")
    _write_case(tmp_path, "BraTS2021_00002")  # same size
    # make the second case bigger
    seg_path = tmp_path / "BraTS2021_00002" / "BraTS2021_00002_seg.nii.gz"
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    seg[5:8, 5:8, 0:4] = 4
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), seg_path)

    cases = list_cases(str(tmp_path))

    assert [c["case_id"] for c in cases] == ["BraTS2021_00002", "BraTS2021_00001"]
    assert cases[0]["volumes_cm3"]["total"] > cases[1]["volumes_cm3"]["total"]
    assert small in {c["path"] for c in cases}


def test_slice_payload_returns_base64_pngs(tmp_path) -> None:
    """A slice returns base64 PNGs for the base image and the 3-region mask."""
    case_dir = _write_case(tmp_path)

    payload = slice_payload(case_dir, z=1)

    base = base64.b64decode(payload["base"])
    mask = base64.b64decode(payload["mask"])
    assert base.startswith(b"\x89PNG")
    assert mask.startswith(b"\x89PNG")
    assert payload["slice_index"] == 1
    assert payload["slices"] == 4
