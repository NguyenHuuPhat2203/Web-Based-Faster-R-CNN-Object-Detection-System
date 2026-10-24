"""Unit tests for authentication service logic."""

import base64
import hashlib

import pytest

from backend.core import security
from backend.core.exceptions import (
    ConflictError,
    OAuthError,
    TokenError,
    UnauthorizedError,
)
from backend.models import TokenBlacklist
from backend.services.auth_service import AuthService
from backend.services.token_service import TokenService


def _google_profile(**overrides) -> dict:
    """Return a verified Google userinfo payload."""
    profile = {
        "id": "g-1",
        "email": "Person@Example.com",
        "name": "Person",
        "verified_email": True,
    }
    return profile | overrides


def _stub_google(monkeypatch, profile: dict) -> None:
    """Replace the Google HTTP calls with a fixed profile."""
    monkeypatch.setattr(
        security, "exchange_google_auth_code", lambda *_: {"access_token": "gt"}
    )
    monkeypatch.setattr(security, "get_google_user_info", lambda _: profile)


class TestAuthService:
    """Auth service unit tests."""

    def test_register_duplicate_email(self, db_session) -> None:
        """Raise when registering a duplicate email."""
        service = AuthService(db_session)
        service.register("dup@example.com", "UserA", "secret123")
        try:
            service.register("dup@example.com", "UserB", "secret123")
            raise AssertionError("Expected ConflictError")
        except ConflictError:
            pass

    def test_authenticate_wrong_password(self, db_session) -> None:
        """Raise when the password is invalid."""
        service = AuthService(db_session)
        service.register("auth@example.com", "User", "secret123")
        try:
            service.authenticate("auth@example.com", "wrong")
            raise AssertionError("Expected UnauthorizedError")
        except UnauthorizedError:
            pass

    def test_email_is_case_insensitive(self, db_session) -> None:
        """Duplicate detection and login ignore email case and whitespace."""
        service = AuthService(db_session)
        service.register("case@example.com", "User", "secret123")
        with pytest.raises(ConflictError):
            service.register("CASE@example.com", "Other", "secret123")
        assert (
            service.authenticate(" Case@Example.com ", "secret123").username == "User"
        )

    def test_overlong_password_never_authenticates(self, db_session) -> None:
        """Passwords above the bcrypt 72-byte limit are rejected, not truncated."""
        service = AuthService(db_session)
        service.register("long@example.com", "User", "x" * 72)
        with pytest.raises(UnauthorizedError):
            service.authenticate("long@example.com", "x" * 72 + "y")


class TestRefreshRotation:
    """Refresh tokens are single-use and typed."""

    def test_refresh_token_is_single_use(self, db_session) -> None:
        """Replaying a refresh token after rotation is rejected."""
        service = AuthService(db_session)
        tokens = service.register("rot@example.com", "User", "secret123")
        service.refresh(tokens.refresh_token)
        with pytest.raises(TokenError, match="revoked"):
            service.refresh(tokens.refresh_token)

    def test_access_token_cannot_refresh(self, db_session) -> None:
        """An access token is rejected by the refresh flow."""
        service = AuthService(db_session)
        tokens = service.register("typ@example.com", "User", "secret123")
        with pytest.raises(TokenError, match="expected refresh"):
            service.refresh(tokens.access_token)

    def test_revoked_expiry_round_trips_as_aware_utc(self, db_session) -> None:
        """Blacklist expiry reads back timezone-aware, so comparisons never mix."""
        tokens = AuthService(db_session).register("tz@example.com", "U", "secret123")
        TokenService(db_session).revoke_tokens(
            tokens.access_token, tokens.refresh_token
        )
        rows = db_session.query(TokenBlacklist).all()
        assert len(rows) == 2
        assert all(row.expires_at.tzinfo is not None for row in rows)
        assert all(row.expires_at > security.utcnow() for row in rows)


class TestGoogleLogin:
    """Google sign-in account matching."""

    def test_creates_user_with_normalized_email(self, db_session, monkeypatch) -> None:
        """New Google identity registers a lowercase-email account."""
        _stub_google(monkeypatch, _google_profile())
        user = AuthService(db_session).google_login_or_register("code", "verifier")
        assert user.email == "person@example.com"

    def test_returns_same_user_for_same_google_id(
        self, db_session, monkeypatch
    ) -> None:
        """Second login by Google ID reuses the account."""
        _stub_google(monkeypatch, _google_profile())
        service = AuthService(db_session)
        first = service.google_login_or_register("code", "verifier")
        second = service.google_login_or_register("code", "verifier")
        assert first.id == second.id

    def test_unverified_email_rejected(self, db_session, monkeypatch) -> None:
        """Unverified Google emails never create or match an account."""
        _stub_google(monkeypatch, _google_profile(verified_email=False))
        with pytest.raises(OAuthError, match="not verified"):
            AuthService(db_session).google_login_or_register("code", "verifier")

    def test_existing_local_account_is_not_linked(
        self, db_session, monkeypatch
    ) -> None:
        """A pre-registered local account with the same email is not taken over."""
        service = AuthService(db_session)
        service.register("person@example.com", "Local", "secret123")
        _stub_google(monkeypatch, _google_profile())
        with pytest.raises(ConflictError):
            service.google_login_or_register("code", "verifier")


class TestOAuthTransaction:
    """PKCE and signed OAuth transaction cookie behavior."""

    def test_create_pkce_pair_uses_s256(self) -> None:
        """Challenge is the unpadded S256 digest of the verifier."""
        verifier, challenge = security.create_pkce_pair()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert challenge == expected
        other, _ = security.create_pkce_pair()
        assert other != verifier

    def test_oauth_tx_round_trip(self) -> None:
        """Signed cookie restores the original state and verifier."""
        token = security.dump_oauth_tx("st", "verifier")
        state, verifier = security.load_oauth_tx(token)
        assert state == "st"
        assert verifier == "verifier"

    def test_oauth_tx_rejects_tampered_signature(self) -> None:
        """Altered HMAC is rejected as invalid OAuth state."""
        token = security.dump_oauth_tx("st", "verifier")
        payload, sig = token.rsplit(".", 1)
        flipped = "0" if sig[0] != "0" else "1"
        tampered = f"{payload}.{flipped}{sig[1:]}"
        with pytest.raises(OAuthError, match="Invalid OAuth state"):
            security.load_oauth_tx(tampered)

    def test_oauth_tx_rejects_empty_and_expired(self, monkeypatch) -> None:
        """Missing cookie and expired payload fail closed."""
        with pytest.raises(OAuthError, match="Invalid OAuth state"):
            security.load_oauth_tx("")
        monkeypatch.setattr(security.time, "time", lambda: 1_000_000.0)
        token = security.dump_oauth_tx("st", "v")
        monkeypatch.setattr(security.time, "time", lambda: 1_000_601.0)
        with pytest.raises(OAuthError, match="Invalid OAuth state"):
            security.load_oauth_tx(token)

    def test_google_auth_url_binds_state_and_pkce(self, db_session) -> None:
        """Consent URL carries state/S256 and cookie round-trips that state."""
        url, cookie, state = AuthService(db_session).get_google_auth_url()
        assert "code_challenge_method=S256" in url
        assert "code_challenge=" in url
        assert f"state={state}" in url
        loaded_state, _ = security.load_oauth_tx(cookie)
        assert loaded_state == state

    def test_oauth_redirect_includes_state(self, db_session) -> None:
        """Frontend fragment includes the OAuth state."""
        tokens = AuthService(db_session).register(
            "oauth@example.com", "OAuthUser", "secret123"
        )
        redirect = AuthService(db_session).build_oauth_redirect(tokens, "abc")
        assert "state=abc" in redirect
