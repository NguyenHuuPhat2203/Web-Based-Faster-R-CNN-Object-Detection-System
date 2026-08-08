"""Integration tests for v2 item-schema prediction endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestPredictV2:
    """v2 Faster R-CNN predict contract."""

    def test_requires_auth(self, client: TestClient, test_image_bytes: bytes) -> None:
        """Unauthenticated requests must receive HTTP 401."""
        response = client.post(
            "/v2/predict",
            files={"file": ("test.png", test_image_bytes, "image/png")},
        )
        assert response.status_code == 401

    def test_item_schema_ok(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Successful inference returns ``{status, result}`` without envelope."""
        response = client.post(
            "/v2/predict",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "result" in data
        assert "success" not in data
        assert "boxes" in data["result"]
        assert data["result"]["model_type"] == "faster-rcnn"

    def test_explain_query_accepted(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """``explain=true`` is accepted on the v2 endpoint."""
        response = client.post(
            "/v2/predict?explain=true",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredictYoloV2:
    """v2 YOLO predict contract."""

    def test_item_schema_ok(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """YOLO v2 returns item-schema."""
        response = client.post(
            "/v2/predict/yolo",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["model_type"] == "yolov8"
