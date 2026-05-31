"""Tests for the predict endpoint."""

from fastapi.testclient import TestClient


class TestPredict:
    def test_requires_auth(self, client: TestClient):
        resp = client.post("/predict", files={
            "file": ("test.png", b"fake-content", "image/png"),
        })
        assert resp.status_code == 401

    def test_rejects_non_image(self, client: TestClient, auth_headers: dict):
        resp = client.post("/predict", files={
            "file": ("test.txt", b"hello", "text/plain"),
        }, headers=auth_headers)
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_rejects_large_file(self, client: TestClient, auth_headers: dict):
        # Send a file bigger than MAX_UPLOAD_SIZE_MB (default 10 MB)
        big = b"x" * (11 * 1024 * 1024)
        resp = client.post("/predict", files={
            "file": ("big.png", big, "image/png"),
        }, headers=auth_headers)
        assert resp.status_code == 413

    def test_accepts_valid_image(self, client: TestClient, auth_headers: dict,
                                 test_image_bytes: bytes):
        resp = client.post("/predict", files={
            "file": ("scan.png", test_image_bytes, "image/png"),
        }, headers=auth_headers)
        # The model will load (pretrained) and run on the tiny image.
        # It may return empty detections, but the shape should be valid.
        assert resp.status_code == 200
        data = resp.json()
        assert "boxes" in data
        assert "scores" in data
        assert "label_names" in data
