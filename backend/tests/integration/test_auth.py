"""Integration tests for authentication endpoints."""

from fastapi.testclient import TestClient


class TestRegister:
    """Tests for user registration."""

    def test_success(self, client: TestClient) -> None:
        """Register a new user and receive tokens."""
        response = client.post(
            "/auth/register",
            json={
                "email": "new@example.com",
                "username": "NewUser",
                "password": "secret123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["username"] == "NewUser"
        assert "access_token" in data
        assert "refresh_token" in data

    def test_duplicate_email(self, client: TestClient) -> None:
        """Reject duplicate email registration."""
        client.post(
            "/auth/register",
            json={
                "email": "dup@example.com",
                "username": "A",
                "password": "secret123",
            },
        )
        response = client.post(
            "/auth/register",
            json={
                "email": "dup@example.com",
                "username": "B",
                "password": "secret123",
            },
        )
        assert response.status_code == 409

    def test_weak_password(self, client: TestClient) -> None:
        """Reject passwords shorter than six characters."""
        response = client.post(
            "/auth/register",
            json={
                "email": "weak@example.com",
                "username": "Weak",
                "password": "ab",
            },
        )
        assert response.status_code == 422


class TestLogin:
    """Tests for user login."""

    def test_success(self, client: TestClient) -> None:
        """Authenticate a registered user."""
        client.post(
            "/auth/register",
            json={
                "email": "login@example.com",
                "username": "Login",
                "password": "pass123456",
            },
        )
        response = client.post(
            "/auth/login",
            json={"email": "login@example.com", "password": "pass123456"},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_wrong_password(self, client: TestClient) -> None:
        """Reject invalid passwords."""
        client.post(
            "/auth/register",
            json={
                "email": "badpwd@example.com",
                "username": "Bad",
                "password": "pass123456",
            },
        )
        response = client.post(
            "/auth/login",
            json={"email": "badpwd@example.com", "password": "wrongpass"},
        )
        assert response.status_code == 401

    def test_nonexistent_user(self, client: TestClient) -> None:
        """Reject unknown users."""
        response = client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "pass123456"},
        )
        assert response.status_code == 401


class TestLogout:
    """Tests for logout behavior."""

    def test_logout_blacklists_tokens(self, client: TestClient) -> None:
        """Blacklist tokens after logout."""
        client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "username": "T",
                "password": "test123456",
            },
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "test123456"},
        )
        payload = login_response.json()

        response = client.post(
            "/auth/logout",
            json={
                "access_token": payload["access_token"],
                "refresh_token": payload["refresh_token"],
            },
        )
        assert response.status_code == 200

        response = client.get(
            "/images",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
        )
        assert response.status_code == 401


class TestRefresh:
    """Tests for token refresh."""

    def test_refresh_success(self, client: TestClient) -> None:
        """Issue a new token pair from a refresh token."""
        client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "username": "T",
                "password": "test123456",
            },
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "test123456"},
        )
        payload = login_response.json()
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": payload["refresh_token"]},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "refresh_token" in response.json()

    def test_refresh_with_bad_token(self, client: TestClient) -> None:
        """Reject invalid refresh tokens."""
        response = client.post("/auth/refresh", json={"refresh_token": "garbage"})
        assert response.status_code == 401
