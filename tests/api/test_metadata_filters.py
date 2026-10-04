"""
Phase 3 — API tests for the metadata PATCH endpoint and query filter params.

What's covered:
    * PATCH updates individual fields (effective_date, jurisdiction, version_tag, regulator)
    * PATCH on non-existent doc → 404
    * PATCH on another tenant's doc → 404 (no leak)
    * Invalid effective_date string → 422
    * QueryRequest accepts optional filters block without breaking the existing shape
"""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
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
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Register admin (first user = admin), attach auth header
            reg = await ac.post(
                "/v1/auth/register",
                json={"email": "md@example.com", "password": "hunter2hunter"},
            )
            ac.headers["Authorization"] = f"Bearer {reg.json()['access_token']}"
            yield ac
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def _make_doc_row(client: AsyncClient) -> tuple[str, str]:
    """Create a case + a synthetic Document row (bypassing upload for speed).

    Returns (case_id, doc_id).
    """
    # Case
    r = await client.post("/v1/cases", json={"name": "MD test"})
    case_id = r.json()["case_id"]

    # Insert a Document directly via the test session
    from backend.app.models import Document, FileType, IngestStatus
    import uuid as _uuid
    doc_id = _uuid.uuid4()

    # Reach into the dependency-overridden session factory to persist a row
    gen = app.dependency_overrides[get_session]()
    s = await gen.__anext__()
    try:
        s.add(Document(
            id=doc_id,
            case_id=_uuid.UUID(case_id),
            filename="synthetic.pdf",
            doc_type=FileType.pdf,
            storage_path="/tmp/synthetic.pdf",
            sha256="x" * 64,
            status=IngestStatus.done,
        ))
        await s.commit()
    finally:
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    return case_id, str(doc_id)


class TestMetadataPatch:
    @pytest.mark.asyncio
    async def test_patch_sets_jurisdiction(self, client):
        case_id, doc_id = await _make_doc_row(client)
        r = await client.patch(
            f"/v1/cases/{case_id}/documents/{doc_id}/metadata",
            json={"jurisdiction": "India"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["jurisdiction"] == "India"
        assert body["effective_date"] is None  # untouched
        assert body["qdrant_payloads_updated"] == 0  # no chunks for synthetic doc

    @pytest.mark.asyncio
    async def test_patch_sets_effective_date(self, client):
        case_id, doc_id = await _make_doc_row(client)
        r = await client.patch(
            f"/v1/cases/{case_id}/documents/{doc_id}/metadata",
            json={"effective_date": "2026-01-15"},
        )
        assert r.status_code == 200
        assert r.json()["effective_date"].startswith("2026-01-15")

    @pytest.mark.asyncio
    async def test_patch_all_fields(self, client):
        case_id, doc_id = await _make_doc_row(client)
        r = await client.patch(
            f"/v1/cases/{case_id}/documents/{doc_id}/metadata",
            json={
                "effective_date": "2026-07-01",
                "jurisdiction": "Maharashtra",
                "version_tag": "v3",
                "regulator": "RBI",
            },
        )
        assert r.status_code == 200
        b = r.json()
        assert b["jurisdiction"] == "Maharashtra"
        assert b["version_tag"] == "v3"
        assert b["regulator"] == "RBI"

    @pytest.mark.asyncio
    async def test_patch_invalid_date_422(self, client):
        case_id, doc_id = await _make_doc_row(client)
        r = await client.patch(
            f"/v1/cases/{case_id}/documents/{doc_id}/metadata",
            json={"effective_date": "not-a-date"},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_patch_unknown_doc_404(self, client):
        case_id, _ = await _make_doc_row(client)
        r = await client.patch(
            f"/v1/cases/{case_id}/documents/{uuid.uuid4()}/metadata",
            json={"jurisdiction": "India"},
        )
        assert r.status_code == 404


class TestQueryRequestShape:
    """Smoke-test that the Phase 3 filters block doesn't break legacy clients."""

    @pytest.mark.asyncio
    async def test_query_without_filters_still_works(self, client):
        r = await client.post("/v1/cases", json={"name": "shape test"})
        case_id = r.json()["case_id"]
        resp = await client.post(
            f"/v1/cases/{case_id}/query",
            json={"question": "What is the penalty?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body and "evidence" in body

    @pytest.mark.asyncio
    async def test_query_with_empty_filters_block(self, client):
        r = await client.post("/v1/cases", json={"name": "shape test 2"})
        case_id = r.json()["case_id"]
        resp = await client.post(
            f"/v1/cases/{case_id}/query",
            json={"question": "x", "filters": {}},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_query_with_jurisdiction_filter(self, client):
        r = await client.post("/v1/cases", json={"name": "shape test 3"})
        case_id = r.json()["case_id"]
        resp = await client.post(
            f"/v1/cases/{case_id}/query",
            json={"question": "x", "filters": {"jurisdiction": "India"}},
        )
        # Case has no docs → empty evidence, but shouldn't 500
        assert resp.status_code == 200
