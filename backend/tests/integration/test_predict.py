"""Integration tests for the Faster R-CNN predict endpoint."""

from fastapi.testclient import TestClient


class TestPredict:
    """Tests for Faster R-CNN prediction."""

    def test_requires_auth(self, client: TestClient) -> None:
        """Reject unauthenticated prediction requests."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", b"fake-content", "image/png")},
        )
        assert response.status_code == 401

    def test_rejects_non_image(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Reject non-image uploads."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_rejects_large_file(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Reject uploads larger than the configured limit."""
        big = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/predict",
            files={"file": ("big.png", big, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 413

    def test_accepts_valid_image(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Accept a valid PNG upload and return prediction fields."""
        response = client.post(
            "/predict",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "boxes" in data
        assert "scores" in data
        assert "label_names" in data
