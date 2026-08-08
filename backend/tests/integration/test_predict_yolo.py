"""Integration tests for the YOLOv8 predict endpoint."""

from fastapi.testclient import TestClient


class TestPredictYolo:
    """Tests for YOLOv8 prediction."""

    def test_requires_auth(self, client: TestClient) -> None:
        """Reject unauthenticated YOLO prediction requests."""
        response = client.post(
            "/predict-yolo",
            files={"file": ("test.png", b"fake-content", "image/png")},
        )
        assert response.status_code == 401

    def test_accepts_valid_image(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Accept a valid PNG upload and return prediction fields."""
        response = client.post(
            "/predict-yolo",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "yolov8"
        assert "boxes" in data
