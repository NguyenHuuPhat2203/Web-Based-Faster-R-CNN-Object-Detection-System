"""Tests for image CRUD endpoints."""

from fastapi.testclient import TestClient


class TestImages:
    def test_list_empty(self, client: TestClient, auth_headers: dict):
        resp = client.get("/images", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"images": []}

    def test_list_after_upload(self, client: TestClient, auth_headers: dict,
                               test_image_bytes: bytes):
        # Upload via predict
        client.post("/predict", files={
            "file": ("scan.png", test_image_bytes, "image/png"),
        }, headers=auth_headers)

        # Now list
        resp = client.get("/images", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 1
        assert data["images"][0]["original_name"] == "scan.png"

    def test_delete_image(self, client: TestClient, auth_headers: dict,
                          test_image_bytes: bytes):
        # Upload
        client.post("/predict", files={
            "file": ("scan.png", test_image_bytes, "image/png"),
        }, headers=auth_headers)

        list_resp = client.get("/images", headers=auth_headers)
        image_id = list_resp.json()["images"][0]["id"]

        # Delete
        del_resp = client.delete(f"/images/{image_id}", headers=auth_headers)
        assert del_resp.status_code == 200

        # Verify gone from list
        list_resp2 = client.get("/images", headers=auth_headers)
        assert list_resp2.json()["images"] == []

    def test_other_user_cannot_delete(self, client: TestClient,
                                      auth_headers: dict, test_image_bytes: bytes):
        # Upload as user 1
        client.post("/predict", files={
            "file": ("scan.png", test_image_bytes, "image/png"),
        }, headers=auth_headers)

        list_resp = client.get("/images", headers=auth_headers)
        image_id = list_resp.json()["images"][0]["id"]

        # Register user 2
        client.post("/auth/register", json={
            "email": "other@example.com", "username": "Other", "password": "pass123456",
        })
        login_resp = client.post("/auth/login", json={
            "email": "other@example.com", "password": "pass123456",
        })
        other_headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

        # Try to delete user 1's image as user 2
        resp = client.delete(f"/images/{image_id}", headers=other_headers)
        assert resp.status_code == 403
