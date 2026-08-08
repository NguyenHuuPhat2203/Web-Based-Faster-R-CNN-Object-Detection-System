"""Integration tests for image CRUD endpoints."""

from fastapi.testclient import TestClient


class TestImages:
    """Tests for image listing and deletion."""

    def test_list_empty(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        """Return an empty list before any uploads."""
        response = client.get("/images", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {"images": []}

    def test_list_after_upload(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Persist uploads through prediction and list them afterward."""
        client.post(
            "/predict",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )

        response = client.get("/images", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 1
        assert data["images"][0]["original_name"] == "scan.png"

    def test_delete_image(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Soft-delete an owned image."""
        client.post(
            "/predict",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )

        list_response = client.get("/images", headers=auth_headers)
        image_id = list_response.json()["images"][0]["id"]

        delete_response = client.delete(f"/images/{image_id}", headers=auth_headers)
        assert delete_response.status_code == 200

        list_response_2 = client.get("/images", headers=auth_headers)
        assert list_response_2.json()["images"] == []

    def test_other_user_cannot_delete(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Prevent cross-user image deletion."""
        client.post(
            "/predict",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )

        list_response = client.get("/images", headers=auth_headers)
        image_id = list_response.json()["images"][0]["id"]

        client.post(
            "/auth/register",
            json={
                "email": "other@example.com",
                "username": "Other",
                "password": "pass123456",
            },
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "other@example.com", "password": "pass123456"},
        )
        other_headers = {
            "Authorization": f"Bearer {login_response.json()['access_token']}"
        }

        response = client.delete(f"/images/{image_id}", headers=other_headers)
        assert response.status_code == 403
