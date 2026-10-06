"""
UI citation preview — tests for GET /v1/cases/{cid}/documents/{did}/page-image.

Covers: auth guard, PDF-only guard, happy path renders PNG, disk cache is used
on the second call, out-of-range page returns 404, deleting the document purges
its cached pages.
"""

import tempfile
import uuid as _uuid
from pathlib import Path

import pymupdf
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.database import get_session
from backend.app.main import app
from backend.app.routers.documents import PAGE_IMAGE_RENDER_SCALE, _PAGE_CACHE_DIR

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
            reg = await ac.post(
                "/v1/auth/register",
                json={"email": "pi@example.com", "password": "hunter2hunter"},
            )
            ac.headers["Authorization"] = f"Bearer {reg.json()['access_token']}"
            yield ac
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _write_pdf(num_pages: int = 2) -> str:
    """Create a tiny 2-page PDF on disk; return its path."""
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page()  # default A4
        page.insert_text((72, 100), f"Page {i + 1} content.", fontsize=14)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(doc.tobytes())
    tmp.close()
    doc.close()
    return tmp.name


async def _make_pdf_doc(client: AsyncClient, pdf_path: str) -> tuple[str, str]:
    """Create a case + a Document row pointing at an on-disk PDF."""
    from backend.app.models import Document, FileType, IngestStatus

    r = await client.post("/v1/cases", json={"name": "PI test"})
    case_id = r.json()["case_id"]
    doc_id = _uuid.uuid4()

    gen = app.dependency_overrides[get_session]()
    s = await gen.__anext__()
    try:
        s.add(Document(
            id=doc_id,
            case_id=_uuid.UUID(case_id),
            filename="synthetic.pdf",
            doc_type=FileType.pdf,
            storage_path=pdf_path,
            sha256="x" * 64,
            status=IngestStatus.done,
            page_count=2,
        ))
        await s.commit()
    finally:
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    return case_id, str(doc_id)


async def _make_audio_doc(client: AsyncClient) -> tuple[str, str]:
    from backend.app.models import Document, FileType, IngestStatus

    r = await client.post("/v1/cases", json={"name": "PI audio"})
    case_id = r.json()["case_id"]
    doc_id = _uuid.uuid4()

    gen = app.dependency_overrides[get_session]()
    s = await gen.__anext__()
    try:
        s.add(Document(
            id=doc_id,
            case_id=_uuid.UUID(case_id),
            filename="voice.mp3",
            doc_type=FileType.audio,
            storage_path="/tmp/voice.mp3",
            sha256="y" * 64,
            status=IngestStatus.done,
        ))
        await s.commit()
    finally:
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
    return case_id, str(doc_id)


class TestPageImage:
    @pytest.mark.asyncio
    async def test_render_scale_constant_is_two(self):
        assert PAGE_IMAGE_RENDER_SCALE == 2

    @pytest.mark.asyncio
    async def test_requires_auth(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get(
                f"/v1/cases/{_uuid.uuid4()}/documents/{_uuid.uuid4()}/page-image?page=1"
            )
            assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_happy_path_returns_png(self, client):
        pdf_path = _write_pdf(2)
        try:
            case_id, doc_id = await _make_pdf_doc(client, pdf_path)
            r = await client.get(
                f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=1"
            )
            assert r.status_code == 200, r.text
            assert r.headers["content-type"] == "image/png"
            assert r.headers.get("x-render-scale") == str(PAGE_IMAGE_RENDER_SCALE)
            # PNG magic bytes
            assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
            # Cached to disk
            cache_file = _PAGE_CACHE_DIR / f"{doc_id}_p1.png"
            assert cache_file.exists()
            cache_file.unlink(missing_ok=True)
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cache_hit_second_call(self, client):
        pdf_path = _write_pdf(2)
        try:
            case_id, doc_id = await _make_pdf_doc(client, pdf_path)
            r1 = await client.get(
                f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=2"
            )
            assert r1.status_code == 200
            # Delete the underlying PDF — a cache miss would now 410. Cache
            # hit should still succeed since the PNG is on disk.
            Path(pdf_path).unlink(missing_ok=True)
            r2 = await client.get(
                f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=2"
            )
            assert r2.status_code == 200
            assert r1.content == r2.content
            (_PAGE_CACHE_DIR / f"{doc_id}_p2.png").unlink(missing_ok=True)
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_page_out_of_range_returns_404(self, client):
        pdf_path = _write_pdf(2)
        try:
            case_id, doc_id = await _make_pdf_doc(client, pdf_path)
            r = await client.get(
                f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=99"
            )
            assert r.status_code == 404
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_non_pdf_returns_400(self, client):
        case_id, doc_id = await _make_audio_doc(client)
        r = await client.get(
            f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=1"
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_doc_returns_404(self, client):
        r = await client.post("/v1/cases", json={"name": "empty"})
        case_id = r.json()["case_id"]
        r = await client.get(
            f"/v1/cases/{case_id}/documents/{_uuid.uuid4()}/page-image?page=1"
        )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_page_param_must_be_positive(self, client):
        pdf_path = _write_pdf(1)
        try:
            case_id, doc_id = await _make_pdf_doc(client, pdf_path)
            r = await client.get(
                f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=0"
            )
            assert r.status_code == 422
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_delete_doc_purges_cached_pages(self, client):
        pdf_path = _write_pdf(2)
        try:
            case_id, doc_id = await _make_pdf_doc(client, pdf_path)
            # Populate cache
            await client.get(f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=1")
            await client.get(f"/v1/cases/{case_id}/documents/{doc_id}/page-image?page=2")
            assert (_PAGE_CACHE_DIR / f"{doc_id}_p1.png").exists()
            assert (_PAGE_CACHE_DIR / f"{doc_id}_p2.png").exists()

            r = await client.delete(f"/v1/cases/{case_id}/documents/{doc_id}")
            assert r.status_code == 200
            assert not (_PAGE_CACHE_DIR / f"{doc_id}_p1.png").exists()
            assert not (_PAGE_CACHE_DIR / f"{doc_id}_p2.png").exists()
        finally:
            Path(pdf_path).unlink(missing_ok=True)
