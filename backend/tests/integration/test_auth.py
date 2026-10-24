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


class TestGoogleOAuth:
    """Google OAuth start and callback binding."""

    def test_start_returns_state_and_cookie(self, client: TestClient) -> None:
        """Start returns state and sets the oauth_tx cookie."""
        response = client.get("/auth/google")
        assert response.status_code == 200
        data = response.json()
        assert data["state"]
        assert "code_challenge" in data["redirect_url"]
        assert "code_challenge_method=S256" in data["redirect_url"]
        assert f"state={data['state']}" in data["redirect_url"]
        cookie = response.headers.get("set-cookie", "")
        assert "oauth_tx=" in cookie
        assert "HttpOnly" in cookie

    def test_callback_without_cookie_is_401(self, client: TestClient) -> None:
        """Callback without the transaction cookie is rejected."""
        response = client.get(
            "/auth/google/callback",
            params={"code": "x", "state": "y"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid OAuth state"

    def test_callback_mismatched_state_is_401(self, client: TestClient) -> None:
        """Callback state that does not match the cookie is rejected."""
        start = client.get("/auth/google")
        assert start.json()["state"] != "not-the-state"
        response = client.get(
            "/auth/google/callback",
            params={"code": "x", "state": "not-the-state"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid OAuth state"


def _register(client: TestClient, email: str = "tok@example.com") -> dict:
    """Register a user and return the token payload."""
    response = client.post(
        "/auth/register",
        json={"email": email, "username": "Tok", "password": "test123456"},
    )
    assert response.status_code == 200
    return response.json()


class TestTokenHandling:
    """Token type, rotation, and revocation behavior over HTTP."""

    def test_refresh_token_rejected_as_access_token(self, client: TestClient) -> None:
        """A refresh token cannot authenticate protected routes."""
        payload = _register(client)
        response = client.get(
            "/images", headers={"Authorization": f"Bearer {payload['refresh_token']}"}
        )
        assert response.status_code == 401

    def test_refresh_token_is_single_use(self, client: TestClient) -> None:
        """Replaying a rotated refresh token is rejected."""
        payload = _register(client)
        body = {"refresh_token": payload["refresh_token"]}
        assert client.post("/auth/refresh", json=body).status_code == 200
        assert client.post("/auth/refresh", json=body).status_code == 401

    def test_logout_revokes_refresh_token(self, client: TestClient) -> None:
        """After logout the refresh token no longer rotates."""
        payload = _register(client)
        client.post(
            "/auth/logout",
            json={
                "access_token": payload["access_token"],
                "refresh_token": payload["refresh_token"],
            },
        )
        response = client.post(
            "/auth/refresh", json={"refresh_token": payload["refresh_token"]}
        )
        assert response.status_code == 401


class TestAccountValidation:
    """Registration and profile input normalization."""

    def test_login_ignores_email_case(self, client: TestClient) -> None:
        """Login succeeds with a differently-cased email."""
        _register(client, "mixed@example.com")
        response = client.post(
            "/auth/login",
            json={"email": "Mixed@Example.COM", "password": "test123456"},
        )
        assert response.status_code == 200

    def test_register_rejects_bad_email_and_long_password(
        self, client: TestClient
    ) -> None:
        """Malformed emails and passwords over 72 bytes are 422."""
        base = {"username": "U", "password": "test123456"}
        bad_email = client.post("/auth/register", json={**base, "email": "nope"})
        long_password = client.post(
            "/auth/register",
            json={"email": "a@example.com", "username": "U", "password": "x" * 73},
        )
        assert bad_email.status_code == 422
        assert long_password.status_code == 422

    def test_profile_email_conflict_is_case_insensitive(
        self, client: TestClient
    ) -> None:
        """Changing to another user's email in different case is a conflict."""
        _register(client, "first@example.com")
        second = _register(client, "second@example.com")
        response = client.put(
            "/auth/profile",
            json={"email": "FIRST@example.com"},
            headers={"Authorization": f"Bearer {second['access_token']}"},
        )
        assert response.status_code == 409


def test_callback_non_ascii_state_is_401_not_500(client: TestClient) -> None:
    """Non-ASCII state parameter fails closed instead of raising TypeError."""
    client.get("/auth/google")
    response = client.get(
        "/auth/google/callback", params={"code": "x", "state": "\u00e9\u00e9"}
    )
    assert response.status_code == 401
