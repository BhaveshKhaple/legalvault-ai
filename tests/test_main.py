"""
Task 6.1 — FastAPI smoke tests.

Verifies: server starts, /docs renders, /healthz returns ok,
all 5 stub endpoints return 200 and correct JSON shapes.
No database connection needed — stubs return static data.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthAndDocs:
    async def test_healthz_returns_ok(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_openapi_schema_available(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "LegalVault AI"


class TestCaseRoutes:
    async def test_create_case_stub(self, client):
        resp = await client.post("/v1/cases")
        assert resp.status_code == 200
        assert "case_id" in resp.json()

    async def test_list_cases_stub(self, client):
        resp = await client.get("/v1/cases")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_get_case_stub(self, client):
        import uuid
        resp = await client.get(f"/v1/cases/{uuid.uuid4()}")
        assert resp.status_code == 200


class TestDocumentRoutes:
    async def test_upload_stub(self, client):
        import uuid
        resp = await client.post(f"/v1/cases/{uuid.uuid4()}/documents")
        assert resp.status_code == 200
        assert "document_id" in resp.json()

    async def test_status_stub(self, client):
        import uuid
        resp = await client.get(f"/v1/cases/{uuid.uuid4()}/documents/{uuid.uuid4()}/status")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestQueryRoute:
    async def test_query_stub(self, client):
        import uuid
        resp = await client.post(
            f"/v1/cases/{uuid.uuid4()}/query",
            json={"question": "What is the penalty clause?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "evidence" in body
        assert "confidence" in body


class TestShieldRoute:
    async def test_shield_stub(self, client):
        import uuid
        resp = await client.post(
            f"/v1/cases/{uuid.uuid4()}/shield",
            json={"source_doc_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "trust_score" in body
        assert "flags" in body


class TestAuthRoute:
    async def test_login_stub(self, client):
        resp = await client.post("/v1/auth/login")
        assert resp.status_code == 200
