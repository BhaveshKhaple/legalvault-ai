"""
Task 6.1/6.2 — FastAPI smoke tests.
Task 6.4 — Fixture now registers an admin user and injects Authorization.
"""

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
            json={"email": "smoke@example.com", "password": "hunter2hunter"},
        )
        assert reg.status_code == 201, reg.text
        ac.headers["Authorization"] = f"Bearer {reg.json()['access_token']}"
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


class TestHealthAndDocs:
    async def test_healthz_returns_ok(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_openapi_schema_available(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        assert resp.json()["info"]["title"] == "LegalVault AI"


class TestCaseRoutes:
    async def test_create_case(self, client):
        resp = await client.post("/v1/cases", json={"name": "Smoke Case"})
        assert resp.status_code == 200
        assert "case_id" in resp.json()

    async def test_list_cases(self, client):
        resp = await client.get("/v1/cases")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_get_case_not_found(self, client):
        resp = await client.get(f"/v1/cases/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestDocumentRoutes:
    async def test_upload_requires_file(self, client):
        # Create a real case first so we clear the auth/ownership check
        r = await client.post("/v1/cases", json={"name": "Doc Smoke"})
        case_id = r.json()["case_id"]
        resp = await client.post(f"/v1/cases/{case_id}/documents")
        assert resp.status_code == 422

    async def test_status_unknown_doc(self, client):
        # Unknown case → 404 (ownership check runs first)
        resp = await client.get(
            f"/v1/cases/{uuid.uuid4()}/documents/{uuid.uuid4()}/status"
        )
        assert resp.status_code == 404


class TestQueryRoute:
    async def test_query_real_case(self, client):
        r = await client.post("/v1/cases", json={"name": "Query Smoke"})
        case_id = r.json()["case_id"]
        resp = await client.post(
            f"/v1/cases/{case_id}/query",
            json={"question": "What is the penalty clause?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "evidence" in body
        assert "confidence" in body

    async def test_query_unknown_case(self, client):
        resp = await client.post(
            f"/v1/cases/{uuid.uuid4()}/query",
            json={"question": "test"},
        )
        assert resp.status_code == 404


class TestShieldRoute:
    async def test_shield_unknown_case(self, client):
        resp = await client.post(
            f"/v1/cases/{uuid.uuid4()}/shield",
            json={"source_doc_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404


class TestAuthRoute:
    async def test_me_returns_registered_user(self, client):
        # `client` fixture already registered smoke@example.com as admin
        resp = await client.get("/v1/auth/me")
        assert resp.status_code == 200
        assert resp.json()["email"] == "smoke@example.com"
        assert resp.json()["role"] == "admin"
