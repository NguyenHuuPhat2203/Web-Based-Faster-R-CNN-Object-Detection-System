"""Integration tests for PDF report generation."""

from fastapi.testclient import TestClient


class TestReport:
    """Tests for detection report downloads."""

    def test_generate_report(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        test_image_bytes: bytes,
    ) -> None:
        """Generate a PDF report for a stored scan."""
        client.post(
            "/predict",
            files={"file": ("scan.png", test_image_bytes, "image/png")},
            headers=auth_headers,
        )
        list_response = client.get("/images", headers=auth_headers)
        image_id = list_response.json()["images"][0]["id"]

        response = client.get(f"/images/{image_id}/report", headers=auth_headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
