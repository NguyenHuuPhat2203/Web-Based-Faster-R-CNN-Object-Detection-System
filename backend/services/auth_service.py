"""Authentication and OAuth business logic."""

from __future__ import annotations

from urllib.parse import urlencode

from sqlalchemy.orm import Session

from ..config import get_settings
from ..core import security
from ..core.exceptions import ConflictError, TokenError, UnauthorizedError
from ..repositories.user_repository import UserRepository
from ..schemas.auth import TokenResponse, UserWithTokens
from .token_service import TokenService


class AuthService:
    """Coordinate registration, login, refresh, and Google OAuth flows."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.db = db
        self.users = UserRepository(db)
        self.tokens = TokenService(db)

    def _build_tokens(self, user) -> UserWithTokens:
        """Issue access and refresh tokens for a user."""
        access_token, _, _ = security.create_access_token(user.id)
        refresh_token, _, _ = security.create_refresh_token(user.id)
        return UserWithTokens(
            id=user.id,
            email=user.email,
            username=user.username,
            created_at=user.created_at,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    def register(self, email: str, username: str, password: str) -> UserWithTokens:
        """Register a new local user.

        Args:
            email: Unique email address.
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
        if not user or not user.hashed_password:
            raise UnauthorizedError("Invalid email or password")
        if not security.verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid email or password")
        return self._build_tokens(user)

    def refresh(self, refresh_token: str) -> TokenResponse:
        """Issue a new token pair from a refresh token.

        Args:
            refresh_token: Valid refresh token string.

        Returns:
            New access and refresh tokens.

        Raises:
            TokenError: If the refresh token is invalid or revoked.
            UnauthorizedError: If the user no longer exists.
        """
        payload = security.decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise TokenError("Invalid token type – expected refresh token")

        jti = payload.get("jti")
        self.tokens.ensure_not_blacklisted(jti)

        user = self.users.get_by_id(int(payload["sub"]))
        if not user:
            raise UnauthorizedError("User not found")

        access_token, _, _ = security.create_access_token(user.id)
        new_refresh_token, _, _ = security.create_refresh_token(user.id)
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    def google_login_or_register(self, code: str) -> UserWithTokens:
        """Complete Google OAuth and return application tokens.

        Args:
            code: Google authorization code.

        Returns:
            Existing or newly created user with issued tokens.
        """
        token_data = security.exchange_google_auth_code(code)
        google_user = security.get_google_user_info(token_data["access_token"])

        google_id = google_user["id"]
        email = google_user["email"]
        username = google_user.get("name", email.split("@")[0])

        user = self.users.get_by_google_id_or_email(google_id, email)
        if user:
            if not user.google_id:
                user.google_id = google_id
                self.users.save(user)
        else:
            user = self.users.create(
                email=email,
                username=username,
                google_id=google_id,
            )
        return self._build_tokens(user)

    def get_google_auth_url(self) -> str:
        """Build the Google OAuth consent-screen URL."""
        settings = get_settings()
        params = {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    def build_oauth_redirect(self, tokens: UserWithTokens) -> str:
        """Build the frontend OAuth callback redirect URL."""
        settings = get_settings()
        return (
            f"{settings.frontend_url}/oauth-callback"
            f"#access_token={tokens.access_token}"
            f"&refresh_token={tokens.refresh_token}"
        )
