"""Unit tests for the BraTS case store (I/O, parse, cache behind the seam)."""

from __future__ import annotations

import base64

import nibabel as nib
import numpy as np
import pytest

from backend.services import case_store


def _write_case(root, name: str = "BraTS2021_00001") -> None:
    """Write a tiny synthetic BraTS-style case (t1ce + seg) into ``root``."""
    case_dir = root / name
    case_dir.mkdir(parents=True)

    seg = np.zeros((8, 8, 4), dtype=np.uint8)
    seg[2:5, 2:5, 1] = 1  # NCR: 3*3 = 9 voxels (slice 1)
    seg[1:6, 1:6, 0] = 2  # ED: 5*5 = 25 voxels (slice 0)
    seg[3:6, 3:6, 3] = 4  # ET: 3*3 = 9 voxels (slice 3)

    base = np.zeros((8, 8, 4), dtype=np.uint16)
    base[..., 1] = 200  # visible contrast on slice 1

    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), case_dir / f"{name}_seg.nii.gz")
    nib.save(nib.Nifti1Image(base, affine=np.eye(4)), case_dir / f"{name}_t1ce.nii.gz")


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the store at a synthetic case root and reset caches."""
    case_store.reset_case_store_cache()

    class _Settings:
        case_viewer_dir = str(tmp_path)

    monkeypatch.setattr(case_store, "get_settings", lambda: _Settings())


def test_analyze_computes_region_volumes(tmp_path) -> None:
    """Voxel counts * spacing must yield per-region cm3 volumes."""
    _write_case(tmp_path)

    info = case_store.analyze("BraTS2021_00001")

    assert info["case_id"] == "BraTS2021_00001"
    assert info["slices"] == 4
    # spacing 1mm -> voxel volume 1 mm3 = 0.001 cm3
    assert info["volumes_cm3"]["ncr"] == pytest.approx(9 * 0.001)
    assert info["volumes_cm3"]["ed"] == pytest.approx(25 * 0.001)
    assert info["volumes_cm3"]["et"] == pytest.approx(9 * 0.001)
    assert info["volumes_cm3"]["total"] == pytest.approx(43 * 0.001)


def test_list_cases_sorts_by_tumor_volume_desc(tmp_path) -> None:
    """Largest-tumor cases come first; cases without seg are filtered."""
    _write_case(tmp_path, "BraTS2021_00001")
    _write_case(tmp_path, "BraTS2021_00002")
    seg_path = tmp_path / "BraTS2021_00002" / "BraTS2021_00002_seg.nii.gz"
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    seg[5:8, 5:8, 0:4] = 4
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), seg_path)
    (tmp_path / "no_seg_case").mkdir()
    (tmp_path / "no_seg_case" / "x_t1ce.nii.gz").write_bytes(b"x")

    cases = case_store.list_cases()

    assert [c["case_id"] for c in cases] == ["BraTS2021_00002", "BraTS2021_00001"]
    assert cases[0]["volumes_cm3"]["total"] > cases[1]["volumes_cm3"]["total"]
    assert all("no_seg_case" not in c["case_id"] for c in cases)


def test_slice_payload_returns_base64_pngs(tmp_path) -> None:
    """A slice returns base64 PNGs for the base image and the region mask."""
    _write_case(tmp_path)

    payload = case_store.slice_payload("BraTS2021_00001", z=1)

    base = base64.b64decode(payload["base"])
    mask = base64.b64decode(payload["mask"])
    assert base.startswith(b"\x89PNG")
    assert mask.startswith(b"\x89PNG")
    assert payload["slice_index"] == 1
    assert payload["slices"] == 4


def test_volume_path_resolves_and_raises(tmp_path) -> None:
    """Volume path points at the right file; unknown kind and missing case raise."""
    _write_case(tmp_path)

    seg = case_store.volume_path("BraTS2021_00001", "seg")
    base = case_store.volume_path("BraTS2021_00001", "t1ce")
    assert seg.name == "BraTS2021_00001_seg.nii.gz"
    assert base.name == "BraTS2021_00001_t1ce.nii.gz"

    with pytest.raises(ValueError):
        case_store.volume_path("BraTS2021_00001", "pred")
    with pytest.raises(FileNotFoundError):
        case_store.volume_path("BraTS2021_99999", "t1ce")


def test_put_mask_and_mask_bytes_roundtrip(tmp_path) -> None:
    """A stored mask round-trips to NIfTI bytes and back."""
    _write_case(tmp_path)
    assert case_store.mask_bytes("BraTS2021_00001") is None

    mask = np.zeros((8, 8, 4), dtype=np.uint8)
    mask[2, 2, 1] = 4
    case_store.put_mask("BraTS2021_00001", mask, np.eye(4), 12.5)

    entry = case_store.get_mask("BraTS2021_00001")
    assert entry is not None and entry.seconds == 12.5
    img = nib.Nifti1Image.from_bytes(case_store.mask_bytes("BraTS2021_00001"))
    assert img.get_fdata()[2, 2, 1] == 4
    assert case_store.mask_bytes("BraTS2021_99999") is None


def test_reset_clears_both_caches(tmp_path) -> None:
    """reset_case_store_cache clears parsed cases and predicted masks."""
    _write_case(tmp_path)
    case_store.analyze("BraTS2021_00001")
    case_store.put_mask(
        "BraTS2021_00001", np.zeros((1, 1, 1), np.uint8), np.eye(4), 1.0
    )
    assert case_store.get_mask("BraTS2021_00001") is not None

    case_store.reset_case_store_cache()

    assert case_store.get_mask("BraTS2021_00001") is None
    # parsed cache refills lazily from disk on the next call
    assert case_store.analyze("BraTS2021_00001")["case_id"] == "BraTS2021_00001"


def test_case_dir_rejects_traversal(tmp_path) -> None:
    """Ids that escape the case root are reported as missing cases."""
    secret = tmp_path.parent / "secret_case"
    secret.mkdir()
    (secret / "x_t1ce.nii.gz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        case_store.case_dir("../secret_case")
    with pytest.raises(FileNotFoundError):
        case_store.case_dir(r"..\\secret_case")
    with pytest.raises(FileNotFoundError):
        case_store.volume_path("BraTS2021_00001/../secret_case", "t1ce")
