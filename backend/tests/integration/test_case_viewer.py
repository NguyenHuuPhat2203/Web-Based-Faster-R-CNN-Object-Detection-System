"""Integration tests for the case-viewer API (ticket 22)."""

from __future__ import annotations

import base64

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.api.routers.case_viewer as case_viewer_router


def _make_case(root, name: str = "BraTS2021_00001") -> str:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    seg = np.zeros((8, 8, 4), dtype=np.uint8)
    seg[2:5, 2:5, 1] = 1
    seg[1:6, 1:6, 0] = 2
    seg[3:6, 3:6, 3] = 4
    base = np.zeros((8, 8, 4), dtype=np.uint16)
    base[..., 1] = 200
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), case_dir / f"{name}_seg.nii.gz")
    nib.save(nib.Nifti1Image(base, affine=np.eye(4)), case_dir / f"{name}_t1ce.nii.gz")
    return str(case_dir)


@pytest.fixture
def case_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the router at a synthetic case and reset the store cache."""
    from backend.services import case_store

    root = tmp_path / "cases"
    root.mkdir()
    case_dir = _make_case(root)
    case_store.reset_case_store_cache()

    class _Settings:
        case_viewer_dir = str(root)

    monkeypatch.setattr(case_store, "get_settings", lambda: _Settings())
    return case_dir


def test_requires_auth(client: TestClient, case_dir: str) -> None:
    """Case-viewer endpoints require authentication like everything else."""
    response = client.get("/case-viewer/cases")
    assert response.status_code == 401


def test_lists_cases_with_volumes(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Listing returns analyzed cases with per-region cm3 volumes."""
    response = client.get("/case-viewer/cases", headers=auth_headers)
    assert response.status_code == 200
    cases = response.json()["cases"]
    assert len(cases) == 1
    case = cases[0]
    assert case["case_id"] == "BraTS2021_00001"
    assert case["slices"] == 4
    assert set(case["volumes_cm3"]) == {"ncr", "ed", "et", "total"}
    assert case["volumes_cm3"]["total"] == pytest.approx(43 * 0.001)


def test_slice_returns_base64_pngs(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """A slice request returns base64 PNGs for base image and mask."""
    response = client.get(
        "/case-viewer/cases/BraTS2021_00001/slices/1", headers=auth_headers
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["slice_index"] == 1
    assert base64.b64decode(payload["base"]).startswith(b"\x89PNG")
    assert base64.b64decode(payload["mask"]).startswith(b"\x89PNG")


def test_slice_clamps_out_of_range(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Out-of-range z values are clamped to the volume."""
    response = client.get(
        "/case-viewer/cases/BraTS2021_00001/slices/999", headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["slice_index"] == 3


def test_volume_stream_requires_auth(client: TestClient, case_dir: str) -> None:
    """Volume download requires authentication."""
    response = client.get("/case-viewer/cases/BraTS2021_00001/volume?type=t1ce")
    assert response.status_code == 401


def test_volume_streams_nifti(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Volume endpoint streams the NIfTI file for NiiVue 3D rendering."""
    response = client.get(
        "/case-viewer/cases/BraTS2021_00001/volume?type=t1ce", headers=auth_headers
    )
    assert response.status_code == 200
    # synthetic files are .nii.gz -> gzip magic 1f 8b
    assert response.content[:2] == b"\x1f\x8b"
    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()
    assert "nii" in content_type or "nii" in disposition
