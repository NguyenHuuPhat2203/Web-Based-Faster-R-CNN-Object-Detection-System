"""Auth routes: register, login, logout, refresh, Google OAuth."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
from models import User, TokenBlacklist
from schemas import (
    RegisterRequest,
    LoginRequest,
    RefreshRequest,
    LogoutRequest,
    UserWithTokens,
    TokenResponse,
)
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    exchange_google_auth_code,
    get_google_user_info,
    _now,
)
import config as cfg

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Helper ────────────────────────────────────────────────────────────

def _build_tokens(user: User) -> UserWithTokens:
    access_token, _, _ = create_access_token(user.id)
    refresh_token, _, _ = create_refresh_token(user.id)
    return UserWithTokens(
        id=user.id,
        email=user.email,
        username=user.username,
        created_at=user.created_at,
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Local Register ────────────────────────────────────────────────────

@router.post("/register", response_model=UserWithTokens)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _build_tokens(user)


# ── Local Login ───────────────────────────────────────────────────────

@router.post("/login", response_model=UserWithTokens)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return _build_tokens(user)


# ── Logout (blacklist tokens) ─────────────────────────────────────────

@router.post("/logout")
def logout(body: LogoutRequest, db: Session = Depends(get_db)):
    now = _now()

    for raw_token in (body.access_token, body.refresh_token):
        try:
            payload = decode_token(raw_token)
        except HTTPException:
            continue  # already expired → nothing to blacklist

        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti:
            exists = db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
            if not exists:
                db.add(TokenBlacklist(
                    jti=jti,
                    expires_at=datetime.fromtimestamp(exp, tz=now.tzinfo) if exp else now,
                ))
    db.commit()
    return {"message": "Logged out successfully"}


# ── Refresh ───────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type – expected refresh token",
        )

    jti = payload.get("jti")
    if jti and db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )

    user_id = int(payload["sub"])
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    access_token, _, _ = create_access_token(user.id)
    refresh_token, _, _ = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Google OAuth ──────────────────────────────────────────────────────

@router.get("/google")
def google_login(request: Request):
    """Redirect the user to Google's consent screen."""
    params = {
        "client_id": cfg.GOOGLE_CLIENT_ID,
        "redirect_uri": cfg.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
    }
    google_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return {"redirect_url": google_url}


@router.get("/google/callback")
def google_callback(code: str, db: Session = Depends(get_db)):
    """Handle the OAuth callback from Google."""
    token_data = exchange_google_auth_code(code)
    google_user = get_google_user_info(token_data["access_token"])

    google_id = google_user["id"]
    email = google_user["email"]
    username = google_user.get("name", email.split("@")[0])

    # Upsert
    user = db.query(User).filter(
        (User.google_id == google_id) | (User.email == email)
    ).first()

    if user:
        # Link google_id if logging in with Google for the first time
        if not user.google_id:
            user.google_id = google_id
            db.commit()
    else:
        user = User(
            email=email,
            username=username,
            google_id=google_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    tokens = _build_tokens(user)

    # Redirect the user's browser back to the frontend with tokens in the URL fragment
    frontend_url = "http://localhost:8001"
    redirect = (
        f"{frontend_url}/oauth-callback"
        f"#access_token={tokens.access_token}"
        f"&refresh_token={tokens.refresh_token}"
    )
    return RedirectResponse(url=redirect)
