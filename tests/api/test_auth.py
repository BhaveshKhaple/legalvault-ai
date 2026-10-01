"""
Task 6.4 — Authentication + authorization tests.

Covers:
- Password hash + verify (unit)
- JWT round-trip encode/decode (unit)
- /register: first user is admin, subsequent users are analyst, duplicate email 409
- /login: correct password succeeds, wrong password fails, unknown email fails
- /me: requires valid Bearer token, returns identity
- Missing token → 401 across protected endpoints
- Cross-tenant isolation: user A cannot see user B's cases (404, not 403 — no leak)
- Role gates: viewer cannot create/delete; analyst can; admin bypasses ownership
"""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.jwt import create_access_token, decode_token
from backend.app.auth.security import hash_password, verify_password
from backend.app.database import get_session
from backend.app.main import app
from backend.app.models import User, UserRole

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ─── unit tests ─────────────────────────────────────────────────────────────


class TestPasswordHashing:
    def test_hash_produces_different_output_each_call(self):
        h1 = hash_password("s3cret-p@ss")
        h2 = hash_password("s3cret-p@ss")
        assert h1 != h2  # argon2 salts each hash

    def test_verify_matches_original(self):
        h = hash_password("s3cret-p@ss")
        assert verify_password("s3cret-p@ss", h) is True

    def test_verify_rejects_wrong_password(self):
        h = hash_password("s3cret-p@ss")
        assert verify_password("wrong", h) is False

    def test_empty_password_rejected(self):
        with pytest.raises(ValueError):
            hash_password("")


class TestJWTRoundTrip:
    def test_encode_decode(self):
        uid = uuid.uuid4()
        token = create_access_token(uid, "analyst")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == str(uid)
        assert payload["role"] == "analyst"
        assert "exp" in payload
        assert "iat" in payload

    def test_invalid_token_returns_none(self):
        assert decode_token("not-a-real-token") is None

    def test_tampered_token_rejected(self):
        token = create_access_token(uuid.uuid4(), "admin")
        tampered = token[:-4] + "xxxx"
        assert decode_token(tampered) is None


# ─── integration fixtures ───────────────────────────────────────────────────


@pytest_asyncio.fixture
async def raw_client():
    """Client with in-memory SQLite but NO pre-registered user."""
    engine = create_async_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async def _override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, Session
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


# ─── /register ──────────────────────────────────────────────────────────────


class TestRegister:
    @pytest.mark.asyncio
    async def test_first_user_is_admin(self, raw_client):
        ac, _ = raw_client
        r = await ac.post("/v1/auth/register", json={
            "email": "first@example.com",
            "password": "hunter2hunter",
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["role"] == "admin"
        assert body["access_token"]
        assert body["user_id"]

    @pytest.mark.asyncio
    async def test_second_user_is_analyst(self, raw_client):
        ac, _ = raw_client
        await ac.post("/v1/auth/register", json={"email": "a@example.com", "password": "hunter2hunter"})
        r = await ac.post("/v1/auth/register", json={"email": "b@example.com", "password": "hunter2hunter"})
        assert r.status_code == 201
        assert r.json()["role"] == "analyst"

    @pytest.mark.asyncio
    async def test_duplicate_email_conflicts(self, raw_client):
        ac, _ = raw_client
        await ac.post("/v1/auth/register", json={"email": "dupe@example.com", "password": "hunter2hunter"})
        r = await ac.post("/v1/auth/register", json={"email": "dupe@example.com", "password": "otherpw123"})
        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_short_password_rejected(self, raw_client):
        ac, _ = raw_client
        r = await ac.post("/v1/auth/register", json={"email": "x@example.com", "password": "short"})
        assert r.status_code == 422


# ─── /login ─────────────────────────────────────────────────────────────────


class TestLogin:
    @pytest.mark.asyncio
    async def test_correct_password_succeeds(self, raw_client):
        ac, _ = raw_client
        await ac.post("/v1/auth/register", json={"email": "u@example.com", "password": "hunter2hunter"})
        r = await ac.post("/v1/auth/login", json={"email": "u@example.com", "password": "hunter2hunter"})
        assert r.status_code == 200
        assert r.json()["access_token"]

    @pytest.mark.asyncio
    async def test_wrong_password_401(self, raw_client):
        ac, _ = raw_client
        await ac.post("/v1/auth/register", json={"email": "u@example.com", "password": "hunter2hunter"})
        r = await ac.post("/v1/auth/login", json={"email": "u@example.com", "password": "wrong-pass"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_email_401(self, raw_client):
        ac, _ = raw_client
        r = await ac.post("/v1/auth/login", json={"email": "ghost@example.com", "password": "hunter2hunter"})
        assert r.status_code == 401


# ─── /me + missing token ────────────────────────────────────────────────────


class TestMe:
    @pytest.mark.asyncio
    async def test_returns_current_user(self, raw_client):
        ac, _ = raw_client
        reg = await ac.post("/v1/auth/register", json={"email": "u@example.com", "password": "hunter2hunter"})
        token = reg.json()["access_token"]
        r = await ac.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["email"] == "u@example.com"
        assert r.json()["role"] == "admin"

    @pytest.mark.asyncio
    async def test_missing_header_401(self, raw_client):
        ac, _ = raw_client
        r = await ac.get("/v1/auth/me")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_header_401(self, raw_client):
        ac, _ = raw_client
        r = await ac.get("/v1/auth/me", headers={"Authorization": "Basic abcdef"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_bogus_token_401(self, raw_client):
        ac, _ = raw_client
        r = await ac.get("/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
        assert r.status_code == 401


# ─── cross-tenant isolation ─────────────────────────────────────────────────


class TestMultiTenancy:
    @pytest.mark.asyncio
    async def test_analyst_cannot_see_other_users_case(self, raw_client):
        ac, _ = raw_client
        # Alice is admin (first user)
        _ = await ac.post("/v1/auth/register", json={"email": "alice@example.com", "password": "hunter2hunter"})
        # Bob is analyst
        bob = await ac.post("/v1/auth/register", json={"email": "bob@example.com", "password": "hunter2hunter"})
        # Carol is analyst
        carol = await ac.post("/v1/auth/register", json={"email": "carol@example.com", "password": "hunter2hunter"})

        bob_token = bob.json()["access_token"]
        carol_token = carol.json()["access_token"]

        # Bob creates a case
        r = await ac.post(
            "/v1/cases",
            json={"name": "Bob's confidential case"},
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        assert r.status_code == 200
        bob_case_id = r.json()["case_id"]

        # Carol tries to fetch it — should 404 (do not leak existence)
        r = await ac.get(
            f"/v1/cases/{bob_case_id}",
            headers={"Authorization": f"Bearer {carol_token}"},
        )
        assert r.status_code == 404

        # Carol's list is empty
        r = await ac.get("/v1/cases", headers={"Authorization": f"Bearer {carol_token}"})
        assert r.status_code == 200
        assert r.json() == []

        # Bob's list has 1
        r = await ac.get("/v1/cases", headers={"Authorization": f"Bearer {bob_token}"})
        assert len(r.json()) == 1

    @pytest.mark.asyncio
    async def test_admin_sees_all_cases(self, raw_client):
        ac, _ = raw_client
        admin = await ac.post("/v1/auth/register", json={"email": "admin@example.com", "password": "hunter2hunter"})
        bob = await ac.post("/v1/auth/register", json={"email": "bob@example.com", "password": "hunter2hunter"})

        admin_token = admin.json()["access_token"]
        bob_token = bob.json()["access_token"]

        await ac.post("/v1/cases", json={"name": "Bob case"},
                      headers={"Authorization": f"Bearer {bob_token}"})

        r = await ac.get("/v1/cases", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        assert len(r.json()) == 1


# ─── role gates ─────────────────────────────────────────────────────────────


class TestRoleGates:
    @pytest.mark.asyncio
    async def test_viewer_cannot_create_case(self, raw_client):
        ac, Session = raw_client
        # First user (admin)
        await ac.post("/v1/auth/register", json={"email": "admin@example.com", "password": "hunter2hunter"})
        # Second user, then downgrade to viewer directly in DB
        reg = await ac.post("/v1/auth/register", json={"email": "vw@example.com", "password": "hunter2hunter"})
        token = reg.json()["access_token"]

        async with Session() as s:
            u = (await s.exec(select(User).where(User.email == "vw@example.com"))).first()
            u.role = UserRole.viewer
            s.add(u)
            await s.commit()

        r = await ac.post("/v1/cases", json={"name": "Nope"},
                          headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403
