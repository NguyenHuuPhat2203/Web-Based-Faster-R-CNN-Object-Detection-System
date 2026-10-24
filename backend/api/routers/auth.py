"""Authentication HTTP routes."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ...config import get_settings
from ...core import security
from ...core.database import get_db
from ...core.exceptions import OAuthError
from ...core.rate_limit import enforce_auth_rate_limit
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

OAUTH_COOKIE_NAME = "oauth_tx"
OAUTH_COOKIE_PATH = "/auth/google"
OAUTH_COOKIE_MAX_AGE_SECONDS = 600


@router.post("/register", response_model=UserWithTokens)
def register(
    body: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(enforce_auth_rate_limit)],
) -> UserWithTokens:
    """Register a new local user account."""
    return AuthService(db).register(body.email, body.username, body.password)


@router.post("/login", response_model=UserWithTokens)
def login(
    body: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(enforce_auth_rate_limit)],
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
    """Rotate a refresh token into a new token pair."""
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
def google_login(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(enforce_auth_rate_limit)],
) -> dict[str, str]:
    """Start Google OAuth: return the consent URL and set the transaction cookie."""
    url, cookie, state = AuthService(db).get_google_auth_url()
    response.set_cookie(
        key=OAUTH_COOKIE_NAME,
        value=cookie,
        httponly=True,
        samesite="lax",
        max_age=OAUTH_COOKIE_MAX_AGE_SECONDS,
        path=OAUTH_COOKIE_PATH,
        secure=get_settings().frontend_url.startswith("https"),
    )
    return {"redirect_url": url, "state": state}


@router.get("/google/callback")
def google_callback(
    code: str,
    state: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(enforce_auth_rate_limit)],
) -> RedirectResponse:
    """Finish Google OAuth and redirect to the frontend with application tokens."""
    raw_cookie = request.cookies.get(OAUTH_COOKIE_NAME)
    expected_state, verifier = security.load_oauth_tx(raw_cookie or "")
    # Compare bytes: str comparison raises TypeError on non-ASCII input.
    if not hmac.compare_digest(expected_state.encode(), state.encode()):
        raise OAuthError("Invalid OAuth state")
    service = AuthService(db)
    tokens = service.google_login_or_register(code, verifier)
    redirect = RedirectResponse(url=service.build_oauth_redirect(tokens, state))
    redirect.delete_cookie(OAUTH_COOKIE_NAME, path=OAUTH_COOKIE_PATH)
    return redirect
