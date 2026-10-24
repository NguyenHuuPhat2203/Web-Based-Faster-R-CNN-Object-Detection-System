"""Authentication and OAuth business logic."""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

from sqlalchemy.orm import Session

from ..config import get_settings
from ..core import security
from ..core.exceptions import ConflictError, OAuthError, TokenError, UnauthorizedError
from ..models import User
from ..repositories.user_repository import UserRepository
from ..schemas.auth import MAX_USERNAME_LENGTH, TokenResponse, UserWithTokens
from .token_service import TokenService

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"


class AuthService:
    """Coordinate registration, login, refresh, and Google OAuth flows."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.db = db
        self.users = UserRepository(db)
        self.tokens = TokenService(db)

    def _build_tokens(self, user: User) -> UserWithTokens:
        """Issue access and refresh tokens for a user."""
        return UserWithTokens(
            id=user.id,
            email=user.email,
            username=user.username,
            created_at=user.created_at,
            access_token=security.create_access_token(user.id),
            refresh_token=security.create_refresh_token(user.id),
        )

    def register(self, email: str, username: str, password: str) -> UserWithTokens:
        """Register a new local user.

        Args:
            email: Unique email address (already normalized by the schema).
            username: Display name.
            password: Plaintext password.

        Returns:
            Created user with issued tokens.

        Raises:
            ConflictError: If the email is already registered.
        """
        if self.users.get_by_email(email):
            raise ConflictError("Email already registered")

        user = self.users.create(
            email=email,
            username=username,
            hashed_password=security.hash_password(password),
        )
        return self._build_tokens(user)

    def authenticate(self, email: str, password: str) -> UserWithTokens:
        """Authenticate a local user.

        Args:
            email: Registered email address.
            password: Plaintext password.

        Returns:
            Authenticated user with issued tokens.

        Raises:
            UnauthorizedError: If credentials are invalid.
        """
        user = self.users.get_by_email(email)
        if user is None or not user.hashed_password:
            # Keep response time independent of whether the account exists.
            security.burn_password_check(password)
            raise UnauthorizedError("Invalid email or password")
        if not security.verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid email or password")
        return self._build_tokens(user)

    def refresh(self, refresh_token: str) -> TokenResponse:
        """Rotate a refresh token into a new access and refresh pair.

        The presented refresh token is revoked, so each one works exactly once.

        Args:
            refresh_token: Valid refresh token string.

        Returns:
            New access and refresh tokens.

        Raises:
            TokenError: If the refresh token is invalid, wrong type, or revoked.
            UnauthorizedError: If the user no longer exists.
        """
        payload = security.decode_token(refresh_token, security.REFRESH_TOKEN_TYPE)
        self.tokens.ensure_not_blacklisted(payload["jti"])

        user = self.users.get_by_id(int(payload["sub"]))
        if user is None:
            raise UnauthorizedError("User not found")

        # Atomic consume: a concurrent replay of the same token loses the race.
        if not self.tokens.revoke_payload(payload):
            raise TokenError("Token has been revoked")

        return TokenResponse(
            access_token=security.create_access_token(user.id),
            refresh_token=security.create_refresh_token(user.id),
        )

    def google_login_or_register(self, code: str, code_verifier: str) -> UserWithTokens:
        """Complete Google OAuth and return application tokens.

        Accounts are matched by Google ID only. An existing account with the same
        email is never auto-linked: local registration does not verify email
        ownership, so linking would let a pre-registered attacker keep access.

        Args:
            code: Google authorization code.
            code_verifier: PKCE verifier for the authorization request.

        Returns:
            Existing or newly created user with issued tokens.

        Raises:
            OAuthError: If Google returns an incomplete or unverified profile.
            ConflictError: If another account already uses the Google email.
        """
        token_data = security.exchange_google_auth_code(code, code_verifier)
        google_token = token_data.get("access_token")
        if not google_token:
            raise OAuthError("Failed to exchange Google auth code")
        google_user = security.get_google_user_info(google_token)

        google_id = str(google_user.get("id") or "")
        email = str(google_user.get("email") or "").strip().lower()
        if not google_id or not email:
            raise OAuthError("Google profile is missing id or email")
        if google_user.get("verified_email") is not True:
            raise OAuthError("Google email is not verified")

        user = self.users.get_by_google_id(google_id)
        if user is None:
            if self.users.get_by_email(email):
                raise ConflictError("An account with this email already exists")
            username = (google_user.get("name") or email.split("@")[0]).strip()
            user = self.users.create(
                email=email,
                username=username[:MAX_USERNAME_LENGTH],
                google_id=google_id,
            )
        return self._build_tokens(user)

    def get_google_auth_url(self) -> tuple[str, str, str]:
        """Build the Google OAuth consent-screen URL and transaction cookie.

        Returns:
            Tuple of consent URL, signed transaction cookie value, and state.
        """
        settings = get_settings()
        state = secrets.token_urlsafe(32)
        verifier, challenge = security.create_pkce_pair()
        params = {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
        return url, security.dump_oauth_tx(state, verifier), state

    def build_oauth_redirect(self, tokens: UserWithTokens, state: str) -> str:
        """Build the frontend OAuth callback redirect URL."""
        fragment = urlencode(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "state": state,
            }
        )
        base = get_settings().frontend_url.rstrip("/")
        return f"{base}/oauth-callback#{fragment}"
