"""Task 6.4 — Pydantic request/response models for auth endpoints."""

import uuid
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from backend.app.models import UserRole


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    role: Optional[UserRole] = None  # only admin can set; ignored for self-signup


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    role: UserRole


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    role: UserRole
