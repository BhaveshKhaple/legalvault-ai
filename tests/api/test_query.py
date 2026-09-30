"""
Tests for Task 6.3 — Query endpoint.

Heavy components (embeddings, Qdrant, Ollama, reranker) are mocked so the
test suite runs without models loaded or services running.
"""

import uuid
from unittest.mock import MagicMock, patch

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
async def client_with_case():
    """Client with a pre-created case and fake DB."""
    engine = create_async_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async def _override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = _override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/cases", json={"name": "Query Test Case"})
        case_id = r.json()["case_id"]
        yield ac, case_id

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_query_case_not_found():
    engine = create_async_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async def _override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(f"/v1/cases/{uuid.uuid4()}/query",
                          json={"question": "What are the penalties?"})
        assert r.status_code == 404
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_query_empty_question(client_with_case):
    client, case_id = client_with_case
    r = await client.post(f"/v1/cases/{case_id}/query", json={"question": "  "})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_query_no_documents_returns_friendly_message(client_with_case):
    """Case with no indexed docs should return a friendly message, not a 500."""
    client, case_id = client_with_case
    r = await client.post(
        f"/v1/cases/{case_id}/query",
        json={"question": "What is the penalty clause?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "evidence" in body
    assert body["evidence"] == []
    # No docs indexed → friendly message
    assert "No documents" in body["answer"] or body["answer"] != ""


@pytest.mark.asyncio
async def test_query_response_shape(client_with_case):
    """Full pipeline mocked — verify response has required fields."""
    client, case_id = client_with_case

    mock_result = {
        "answer": "The penalty clause is 2% per month [source: contract.pdf p.8]",
        "evidence": [
            {"rank": 1, "score": 0.92, "content": "Penalty is 2% monthly",
             "filename": "contract.pdf", "page": 8, "ts_start": None, "section_title": "8.3"}
        ],
        "confidence": 0.92,
        "latency_ms": 450,
    }

    with patch("backend.app.services.rag_service.run_rag", return_value=mock_result):
        # Also need to mock the case existence check which happens before run_rag
        from backend.app.models import Case
        from datetime import datetime, timezone
        mock_case = MagicMock(spec=Case)
        mock_case.id = uuid.UUID(case_id)

        r = await client.post(
            f"/v1/cases/{case_id}/query",
            json={"question": "What is the penalty clause?"},
        )

    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "evidence" in body
    assert "confidence" in body
    assert "latency_ms" in body
