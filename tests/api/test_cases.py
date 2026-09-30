"""Tests for Task 6.2 — Cases CRUD endpoints (auth-gated in Task 6.4)."""

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.database import get_session
from backend.app.main import app

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    """Async test client + in-memory SQLite + pre-registered admin user."""
    engine = create_async_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async def _override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = _override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        reg = await ac.post(
            "/v1/auth/register",
            json={"email": "alice@example.com", "password": "hunter2hunter"},
        )
        assert reg.status_code == 201, reg.text
        ac.headers["Authorization"] = f"Bearer {reg.json()['access_token']}"
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_case(client):
    r = await client.post("/v1/cases", json={"name": "Test v. Plaintiff", "doc_type": "contract"})
    assert r.status_code == 200
    data = r.json()
    assert "case_id" in data
    assert data["name"] == "Test v. Plaintiff"


@pytest.mark.asyncio
async def test_list_cases_empty(client):
    r = await client.get("/v1/cases")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_cases_after_create(client):
    await client.post("/v1/cases", json={"name": "Case A"})
    await client.post("/v1/cases", json={"name": "Case B"})
    r = await client.get("/v1/cases")
    assert r.status_code == 200
    assert len(r.json()) == 2


@pytest.mark.asyncio
async def test_get_case(client):
    r = await client.post("/v1/cases", json={"name": "NDA Review"})
    case_id = r.json()["case_id"]
    r2 = await client.get(f"/v1/cases/{case_id}")
    assert r2.status_code == 200
    assert r2.json()["case_id"] == case_id
    assert r2.json()["name"] == "NDA Review"
    assert r2.json()["documents"] == []


@pytest.mark.asyncio
async def test_get_case_not_found(client):
    r = await client.get(f"/v1/cases/{uuid.uuid4()}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_case(client):
    r = await client.post("/v1/cases", json={"name": "To Delete"})
    case_id = r.json()["case_id"]
    r2 = await client.delete(f"/v1/cases/{case_id}")
    assert r2.status_code == 200
    assert r2.json()["deleted"] is True
    r3 = await client.get(f"/v1/cases/{case_id}")
    assert r3.status_code == 404


@pytest.mark.asyncio
async def test_missing_auth_returns_401():
    """No Authorization header → 401 on protected endpoints."""
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
            r = await ac.get("/v1/cases")
            assert r.status_code == 401
            r = await ac.post("/v1/cases", json={"name": "x"})
            assert r.status_code == 401
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()
