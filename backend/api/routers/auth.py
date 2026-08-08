"""Authentication HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...models import User
from ...schemas.auth import (
    LoginRequest,
    LogoutRequest,
    PasswordChangeRequest,
    ProfileUpdateRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
    UserStats,
    UserWithTokens,
)
from ...services.auth_service import AuthService
from ...services.stats_service import StatsService
from ...services.token_service import TokenService
from ...services.user_service import UserService
from ..dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserWithTokens)
def register(
    body: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
) -> UserWithTokens:
    """Register a new local user account."""
    return AuthService(db).register(body.email, body.username, body.password)


@router.post("/login", response_model=UserWithTokens)
def login(
    body: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> UserWithTokens:
    """Authenticate a local user and issue tokens."""
    return AuthService(db).authenticate(body.email, body.password)


@router.post("/logout")
def logout(
    body: LogoutRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Revoke the provided access and refresh tokens."""
    TokenService(db).revoke_tokens(body.access_token, body.refresh_token)
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    body: RefreshRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Issue a new token pair from a refresh token."""
    return AuthService(db).refresh(body.refresh_token)


@router.put("/profile", response_model=UserOut)
def update_profile(
    body: ProfileUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Update the authenticated user's profile."""
    return UserService(db).update_profile(
        current_user,
        username=body.username,
        email=body.email,
    )


@router.put("/password")
def change_password(
    body: PasswordChangeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Change the authenticated user's password."""
    UserService(db).change_password(
        current_user,
        current_password=body.current_password,
        new_password=body.new_password,
    )
    return {"message": "Password updated successfully"}


@router.get("/stats", response_model=UserStats)
def user_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> UserStats:
    """Return aggregated scan statistics for the authenticated user."""
    return StatsService(db).get_user_stats(current_user)


@router.get("/google")
def google_login(db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    """Return the Google OAuth consent-screen URL."""
    return {"redirect_url": AuthService(db).get_google_auth_url()}


@router.get("/google/callback")
def google_callback(
    code: str,
    db: Annotated[Session, Depends(get_db)],
) -> RedirectResponse:
    """Handle the Google OAuth callback and redirect to the frontend."""
    service = AuthService(db)
    tokens = service.google_login_or_register(code)
    return RedirectResponse(url=service.build_oauth_redirect(tokens))
