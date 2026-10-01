"""
Task 6.4 — JWT encode/decode.

Payload:
    sub         subject (user_id, str)
    role        user role snapshot at issue time (informational; DB is authoritative)
    exp         expiry (unix ts)
    iat         issued at (unix ts)

Signed HS256 with JWT_SECRET_KEY from env. In prod set a 256-bit random secret.
Never commit a real secret. `.env.example` ships a placeholder only.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt


_ALGORITHM = "HS256"
_DEFAULT_TTL_MIN = 60 * 12  # 12 hours


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY", "").strip()
    if not secret:
        # Fail loud in prod, but keep dev experience friendly.
        # A weak default so `pytest` and `uvicorn` still boot without a .env.
        secret = "dev-only-insecure-secret-change-me"
    return secret


def _ttl_minutes() -> int:
    try:
        return int(os.environ.get("JWT_TTL_MINUTES", _DEFAULT_TTL_MIN))
    except ValueError:
        return _DEFAULT_TTL_MIN


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=_ttl_minutes())).timestamp()),
    }
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Return decoded payload or None if invalid/expired."""
    try:
        return jwt.decode(token, _secret(), algorithms=[_ALGORITHM])
    except JWTError:
        return None
