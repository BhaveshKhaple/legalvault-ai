"""
Task 6.4 — Auth endpoints: /register, /login, /me.

/register — public. First user in the DB becomes admin automatically; every
subsequent self-signup is 'analyst' (viewer / role escalation requires an
admin call — not exposed yet).

/login    — email + password → JWT (12h TTL by default).

/me       — introspect the caller's identity from their token.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import get_current_user
from backend.app.auth.jwt import create_access_token
from backend.app.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.app.auth.security import hash_password, verify_password
from backend.app.database import get_session
from backend.app.models import User, UserRole

router = APIRouter()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account. First user becomes admin.",
)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    existing = (
        await session.exec(select(User).where(User.email == body.email))
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    total_users = (await session.exec(select(User))).all()
    role = UserRole.admin if not total_users else UserRole.analyst

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        role=role,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    token = create_access_token(user.id, user.role.value)
    return TokenResponse(access_token=token, user_id=user.id, role=user.role)


@router.post("/login", response_model=TokenResponse, summary="Password login → JWT")
async def login(
    body: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    user = (
        await session.exec(select(User).where(User.email == body.email))
    ).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id, user.role.value)
    return TokenResponse(access_token=token, user_id=user.id, role=user.role)


@router.get("/me", response_model=UserResponse, summary="Get the caller's user")
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=user.id, email=user.email, role=user.role)
