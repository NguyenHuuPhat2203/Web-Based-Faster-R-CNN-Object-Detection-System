"""Integration tests for v2 item-schema prediction endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services.prediction_service import PredictionService


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
        listed = client.get("/images", headers=auth_headers).json()["images"]
        assert data["result"]["image_id"] == listed[0]["id"]

    def test_returns_full_box_set_below_default_threshold(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """v2 returns all post-NMS boxes (client slider owns filtering)."""
        response = client.post(
            "/v2/predict",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        result = response.json()["result"]
        # Mock returns scores [0.9, 0.3]; both must be present even though the
        # default threshold is 0.5 — the server must NOT filter for v2.
        assert len(result["boxes"]) == 2
        assert result["scores"] == [0.9, 0.3]

    def test_passes_floor_score(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
        monkeypatch,
    ) -> None:
        """v2 routes the inference floor (0.1) into the service."""
        captured: dict = {}
        original = PredictionService.run

        def spy(self, **kwargs):
            captured.update(kwargs)
            return original(self, **kwargs)

        monkeypatch.setattr(PredictionService, "run", spy)
        response = client.post(
            "/v2/predict",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert captured["score_floor"] == 0.1

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

    def test_sanitizes_inference_exception(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
        monkeypatch,
    ) -> None:
        """Inference exceptions must not leak internals in the response."""

        def boom(self, **kwargs):
            raise RuntimeError("/secret/checkpoint")

        monkeypatch.setattr(PredictionService, "run", boom)
        response = client.post(
            "/v2/predict",
            files={"file": ("test.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.json()["error"] == {
            "code": "INFERENCE_FAILED",
            "message": "Inference failed",
        }
        assert "/secret/checkpoint" not in response.text


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
