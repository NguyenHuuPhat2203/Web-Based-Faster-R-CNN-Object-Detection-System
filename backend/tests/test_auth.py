"""Tests for auth endpoints: register, login, logout, refresh."""

from fastapi.testclient import TestClient


class TestRegister:
    def test_success(self, client: TestClient):
        resp = client.post("/auth/register", json={
            "email": "new@example.com",
            "username": "NewUser",
            "password": "secret123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "new@example.com"
        assert data["username"] == "NewUser"
        assert "access_token" in data
        assert "refresh_token" in data

    def test_duplicate_email(self, client: TestClient):
        client.post("/auth/register", json={
            "email": "dup@example.com", "username": "A", "password": "secret123",
        })
        resp = client.post("/auth/register", json={
            "email": "dup@example.com", "username": "B", "password": "secret123",
        })
        assert resp.status_code == 409

    def test_weak_password(self, client: TestClient):
        resp = client.post("/auth/register", json={
            "email": "weak@example.com", "username": "Weak", "password": "ab",
        })
        assert resp.status_code == 422


class TestLogin:
    def test_success(self, client: TestClient):
        client.post("/auth/register", json={
            "email": "login@example.com", "username": "Login", "password": "pass123456",
        })
        resp = client.post("/auth/login", json={
            "email": "login@example.com", "password": "pass123456",
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_wrong_password(self, client: TestClient):
        client.post("/auth/register", json={
            "email": "badpwd@example.com", "username": "Bad", "password": "pass123456",
        })
        resp = client.post("/auth/login", json={
            "email": "badpwd@example.com", "password": "wrongpass",
        })
        assert resp.status_code == 401

    def test_nonexistent_user(self, client: TestClient):
        resp = client.post("/auth/login", json={
            "email": "ghost@example.com", "password": "pass123456",
        })
        assert resp.status_code == 401


class TestLogout:
    def test_logout_blacklists_tokens(self, client: TestClient):
        # Register + login
        client.post("/auth/register", json={
            "email": "test@example.com", "username": "T", "password": "test123456",
        })
        login_resp = client.post("/auth/login", json={
            "email": "test@example.com", "password": "test123456",
        })
        d = login_resp.json()

        # Logout
        resp = client.post("/auth/logout", json={
            "access_token": d["access_token"],
            "refresh_token": d["refresh_token"],
        })
        assert resp.status_code == 200

        # Using the now-blacklisted token should fail
        resp = client.get("/images", headers={
            "Authorization": f"Bearer {d['access_token']}"
        })
        assert resp.status_code == 401


class TestRefresh:
    def test_refresh_success(self, client: TestClient):
        client.post("/auth/register", json={
            "email": "test@example.com", "username": "T", "password": "test123456",
        })
        login_resp = client.post("/auth/login", json={
            "email": "test@example.com", "password": "test123456",
        })
        d = login_resp.json()
        resp = client.post("/auth/refresh", json={
            "refresh_token": d["refresh_token"],
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()
        assert "refresh_token" in resp.json()

    def test_refresh_with_bad_token(self, client: TestClient):
        resp = client.post("/auth/refresh", json={"refresh_token": "garbage"})
        assert resp.status_code == 401
