"""JWT utilities, password hashing, and Google OAuth helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
import requests
from fastapi import HTTPException, status

import config as cfg

# ── Password ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_jti() -> str:
    return uuid.uuid4().hex


def create_access_token(user_id: int) -> tuple[str, str, datetime]:
    """Returns (token, jti, expires_at)."""
    jti = _make_jti()
    exp = _now() + timedelta(minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "jti": jti,
        "type": "access",
        "exp": exp,
    }
    token = jwt.encode(payload, cfg.SECRET_KEY, algorithm=cfg.ALGORITHM)
    return token, jti, exp


def create_refresh_token(user_id: int) -> tuple[str, str, datetime]:
    """Returns (token, jti, expires_at)."""
    jti = _make_jti()
    exp = _now() + timedelta(days=cfg.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "jti": jti,
        "type": "refresh",
        "exp": exp,
    }
    token = jwt.encode(payload, cfg.SECRET_KEY, algorithm=cfg.ALGORITHM)
    return token, jti, exp


def decode_token(token: str) -> dict:
    """Decode and validate a JWT.  Raises 401 on any failure."""
    try:
        payload = jwt.decode(token, cfg.SECRET_KEY, algorithms=[cfg.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


# ── Google OAuth ──────────────────────────────────────────────────────

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def exchange_google_auth_code(code: str) -> dict:
    """Exchange the one‑time code from Google for access + id tokens."""
    data = {
        "code": code,
        "client_id": cfg.GOOGLE_CLIENT_ID,
        "client_secret": cfg.GOOGLE_CLIENT_SECRET,
        "redirect_uri": cfg.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=data, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to exchange Google auth code",
        )
    return resp.json()


def get_google_user_info(access_token: str) -> dict:
    """Fetch the user's profile from Google."""
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(GOOGLE_USERINFO_URL, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to fetch Google user info",
        )
    return resp.json()
