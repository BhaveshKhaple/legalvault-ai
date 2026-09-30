"""
Task 6.4 — FastAPI dependencies for JWT auth + role gates + case ownership.

Usage:
    @router.get("...")
    async def endpoint(user: User = Depends(get_current_user)):
        ...

    @router.post("...")
    async def endpoint(user: User = Depends(require_role(UserRole.analyst))):
        # user is analyst or admin
        ...

Role hierarchy (higher includes lower): admin > analyst > viewer.
"""

import uuid
from typing import Callable

from fastapi import Depends, Header, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.jwt import decode_token
from backend.app.database import get_session
from backend.app.models import Case, User, UserRole


_ROLE_RANK = {UserRole.viewer: 1, UserRole.analyst: 2, UserRole.admin: 3}


async def get_current_user(
    authorization: str = Header(default=""),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Resolve the caller's User from the Bearer token. 401 if invalid."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = uuid.UUID(payload["sub"])
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def require_role(minimum: UserRole) -> Callable:
    """Dependency factory: 403 if the caller's role is below `minimum`."""
    required_rank = _ROLE_RANK[minimum]

    async def _dep(user: User = Depends(get_current_user)) -> User:
        if _ROLE_RANK.get(user.role, 0) < required_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{minimum.value}' or higher required (you are '{user.role.value}')",
            )
        return user

    return _dep


async def get_case_for_user(
    case_id: uuid.UUID,
    session: AsyncSession,
    user: User,
) -> Case:
    """Fetch a case + enforce ownership. Admin sees all cases.

    Legacy cases with created_by=NULL (created before 6.4) are visible to
    admin only. Analysts/viewers can only see cases they created.
    """
    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if user.role == UserRole.admin:
        return case

    if case.created_by is None or case.created_by != user.id:
        # Return 404, not 403 — do not leak that the case exists
        raise HTTPException(status_code=404, detail="Case not found")

    return case
