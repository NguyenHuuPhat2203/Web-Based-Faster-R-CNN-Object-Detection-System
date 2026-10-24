"""Integration tests for live 3D segmentation via the case-viewer API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.services import case_store, segmentation_service

_CASE_ID = "BraTS2021_00001"
_SEGMENT_URL = f"/case-viewer/cases/{_CASE_ID}/segment"


class _FakeSegmentor:
    """Stand-in for SwinUNETRSegmentor: instant, deterministic mask."""

    calls = 0

    def segment_volume(self, case_dir) -> tuple[np.ndarray, np.ndarray, float]:
        _FakeSegmentor.calls += 1
        mask = np.zeros((8, 8, 4), dtype=np.uint8)
        mask[2:4, 2:4, 1] = 1
        mask[1:6, 1:6, 0] = 2
        mask[4:6, 4:6, 2] = 4
        return mask, np.diag([10.0, 10.0, 10.0, 1.0]), 0.25


@pytest.fixture
def case_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Synthetic case dir, store pointed at it, fake model injected."""
    root = tmp_path / "cases"
    (root / _CASE_ID).mkdir(parents=True)
    case_store.reset_case_store_cache()
    _FakeSegmentor.calls = 0

    class _Settings:
        case_viewer_dir = str(root)

    monkeypatch.setattr(case_store, "get_settings", lambda: _Settings())
    monkeypatch.setattr(segmentation_service, "_segmentor", _FakeSegmentor)
    return root


def test_segment_requires_auth(client: TestClient, case_root) -> None:
    """Segmenting requires authentication."""
    assert client.post(_SEGMENT_URL).status_code == 401


def test_segment_returns_volumes_once(
    client: TestClient, auth_headers: dict[str, str], case_root
) -> None:
    """Segmentation returns region volumes and reuses the stored mask."""
    first = client.post(_SEGMENT_URL, headers=auth_headers)
    second = client.post(_SEGMENT_URL, headers=auth_headers)

    assert first.status_code == 200
    body = first.json()
    assert body["case_id"] == _CASE_ID
    # 10 mm voxels -> 1 cm3 each: NCR 4, ED 25, ET 4
    assert body["volumes_cm3"] == {"NCR": 4.0, "ED": 25.0, "ET": 4.0}
    assert second.json() == body
    assert _FakeSegmentor.calls == 1


def test_segment_missing_case_404(
    client: TestClient, auth_headers: dict[str, str], case_root
) -> None:
    """Unknown case returns 404."""
    response = client.post(
        "/case-viewer/cases/BraTS2021_99999/segment", headers=auth_headers
    )
    assert response.status_code == 404


def test_segment_model_unavailable_503(
    client: TestClient,
    auth_headers: dict[str, str],
    case_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing weights surface as 503, not 404 or 500."""

    def _unavailable():
        raise segmentation_service.SegmentationUnavailableError("no weights")

    monkeypatch.setattr(segmentation_service, "_segmentor", _unavailable)
    assert client.post(_SEGMENT_URL, headers=auth_headers).status_code == 503


def test_pred_volume_after_segment(
    client: TestClient, auth_headers: dict[str, str], case_root
) -> None:
    """Predicted volume is 404 before segmentation and NIfTI after."""
    url = f"/case-viewer/cases/{_CASE_ID}/volume?type=pred"
    assert client.get(url, headers=auth_headers).status_code == 404

    client.post(_SEGMENT_URL, headers=auth_headers)
    response = client.get(url, headers=auth_headers)

    assert response.status_code == 200
    # NIfTI-1 single-file magic 'n+1' sits at header offset 344
    assert response.content[344:347] == b"n+1"
