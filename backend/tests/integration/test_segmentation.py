"""Integration tests for live 3D segmentation (SwinUNETR) via the case-viewer API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.services import segmentation_service

# segment3d (SwinUNETR) temporarily unwired — /segment endpoint removed.
pytestmark = pytest.mark.skip(reason="segment3d temporarily unwired")


class _FakeSegmentor:
    """Stand-in for SwinUNETRSegmentor: instant, deterministic mask."""

    _inst: _FakeSegmentor | None = None

    @classmethod
    def get(cls) -> _FakeSegmentor:
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self) -> None:
        self.mask = np.zeros((8, 8, 4), dtype=np.uint8)
        self.mask[2:4, 2:4, 1] = 1  # NCR
        self.mask[1:6, 1:6, 0] = 2  # ED
        self.mask[4:6, 4:6, 2] = 4  # ET

    def segment_volume(self, case_dir) -> tuple[np.ndarray, np.ndarray, float]:
        return self.mask, np.eye(4), 0.25

    def volumes_cm3(self, mask, affine) -> dict[str, float]:
        return {
            name: float(np.sum(mask == lab))
            for lab, name in [(1, "NCR"), (2, "ED"), (4, "ET")]
        }


@pytest.fixture(autouse=True)
def _fake_segmentor(monkeypatch: pytest.MonkeyPatch) -> None:
    segmentation_service.reset_segmentation_cache()
    monkeypatch.setattr(segmentation_service, "SwinUNETRSegmentor", _FakeSegmentor)


@pytest.fixture
def case_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Synthetic case dir + settings pointed at it."""
    from backend.services import case_store

    root = tmp_path / "cases"
    root.mkdir()
    case = root / "BraTS2021_00001"
    case.mkdir()
    (case / "BraTS2021_00001_t1ce.nii.gz").write_bytes(b"x")
    (case / "BraTS2021_00001_seg.nii.gz").write_bytes(b"x")
    case_store.reset_case_store_cache()

    class _Settings:
        case_viewer_dir = str(root)

    monkeypatch.setattr(
        "backend.api.routers.case_viewer.get_settings", lambda: _Settings()
    )
    monkeypatch.setattr(
        "backend.services.segmentation_service.get_settings", lambda: _Settings()
    )
    return str(case)


def test_segment_requires_auth(client: TestClient, case_dir: str) -> None:
    """Segmenting requires authentication."""
    response = client.post("/case-viewer/cases/BraTS2021_00001/segment")
    assert response.status_code == 401


def test_segment_returns_volumes(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Segmentation returns region volumes."""
    response = client.post(
        "/case-viewer/cases/BraTS2021_00001/segment", headers=auth_headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "BraTS2021_00001"
    assert body["seconds"] >= 0
    assert set(body["volumes_cm3"]) == {"NCR", "ED", "ET"}


def test_segment_cached_second_call(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Repeated segmentation hits the cache (same volumes)."""
    post = lambda: client.post(  # noqa: E731
        "/case-viewer/cases/BraTS2021_00001/segment", headers=auth_headers
    ).json()
    first, second = post(), post()
    assert first["volumes_cm3"] == second["volumes_cm3"]


def test_segment_missing_case_404(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Unknown case returns 404."""
    response = client.post(
        "/case-viewer/cases/BraTS2021_99999/segment", headers=auth_headers
    )
    assert response.status_code == 404


def test_pred_volume_requires_segmentation_first(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Predicted volume is 404 before segmentation runs."""
    response = client.get(
        "/case-viewer/cases/BraTS2021_00001/volume?type=pred", headers=auth_headers
    )
    assert response.status_code == 404


def test_pred_volume_streams_nifti_after_segment(
    client: TestClient, auth_headers: dict[str, str], case_dir: str
) -> None:
    """Predicted volume streams NIfTI after segmentation."""
    response = client.post(
        "/case-viewer/cases/BraTS2021_00001/segment", headers=auth_headers
    )
    assert response.status_code == 200
    response = client.get(
        "/case-viewer/cases/BraTS2021_00001/volume?type=pred", headers=auth_headers
    )
    assert response.status_code == 200
    # NIfTI-1 magic 'n+1' sits at header offset 344 (first 4 bytes = sizeof_hdr)
    assert response.content[344:347] == b"n+1"
