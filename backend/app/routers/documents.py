"""
Task 6.2 — Document upload endpoint with synchronous ingestion pipeline.
embedding-dispatcher — At ingestion time, the dispatcher decides whether
to use e5-small-v2 (384-dim → legalvault_e5) or BGE-M3 (1024-dim →
legalvault_bge). The decision is stored on the Document record so the
RAG service can fan-out to the right collection(s) at query time.

Upload flow:
  1. Receive PDF / TXT / DOCX / audio via multipart POST
  2. Save to data/documents/<doc_id>.<ext>
  3. Extract text (extractor by suffix) or transcribe (whisper)
  4. Run dispatcher → routing decision (e5_small | bge_m3)
  5. Embed with the chosen model, insert into the matching Qdrant collection
  6. Persist Chunk records to SQLite (BM25 corpus at query time)
  7. Stamp embedding_tier + qdrant_collection on Document, set status → done
"""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import (
    get_case_for_user,
    get_current_user,
    require_role,
)
from backend.app.database import get_session
from backend.app.models import (
    Chunk,
    Document,
    FileType,
    IngestStatus,
    User,
    UserRole,
)


def _parse_effective_date(raw: Optional[str]) -> Optional[datetime]:
    """Accept 'YYYY-MM-DD' or full ISO datetime; return tz-aware UTC datetime.

    SQLModel's DateTime column rejects naive datetimes — any value we persist
    must carry a timezone. If the input is a bare date we attach UTC.
    """
    from datetime import timezone

    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=422,
            detail=f"effective_date must be ISO format (YYYY-MM-DD or full ISO), got '{raw}'",
        )


def _metadata_payload(doc: Document) -> dict:
    """Build the metadata subset we stamp on every Qdrant chunk payload.

    Dates serialised as unix timestamp (int) so Qdrant's range filter works.
    Strings serialised as-is (match filter).
    """
    out: dict = {}
    if doc.effective_date is not None:
        out["effective_date_ts"] = int(doc.effective_date.timestamp())
    if doc.jurisdiction:
        out["jurisdiction"] = doc.jurisdiction
    if doc.version_tag:
        out["version_tag"] = doc.version_tag
    if doc.regulator:
        out["regulator"] = doc.regulator
    return out

router = APIRouter()

_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_TEXT_SUFFIXES = {".txt", ".docx"}
_ALLOWED_SUFFIXES = {".pdf"} | _AUDIO_SUFFIXES | _TEXT_SUFFIXES


# ─── sync ingestion helpers (run in thread pool) ─────────────────────────────


def _ingest_text_like(
    path: str,
    doc_id: uuid.UUID,
    case_id: uuid.UUID,
    filename: str,
    suffix: str,
    extra_metadata: Optional[dict] = None,
) -> tuple[dict, str, str, str]:
    """Extract → hierarchical chunk → dispatch → embed children → index to Qdrant.

    Returns:
        ({"parents": [...], "children": [... with qdrant_id]}, qdrant_collection, embedding_tier, extractor_used)

    Only CHILDREN are embedded + indexed to Qdrant. Parents stay in SQLite
    and are swapped in at the LLM-context stage of rag_service.run_rag().
    """
    from backend.app.ingestion.embedding_dispatcher import dispatch
    from backend.app.ingestion.hierarchical_chunker import chunk_hierarchical
    from backend.app.retrieval.embeddings import embed_batch
    from backend.app.retrieval.vector_store import insert

    extractor_used = "unknown"

    # ── 1. Extract pages (Docling for PDF/DOCX, fallbacks for TXT/legacy) ────
    if suffix == ".pdf":
        from backend.app.ingestion.docling_extractor import extract_docling
        from backend.app.ingestion.pdf_extractor import extract_pdf
        try:
            pages = extract_docling(path)
            extractor_used = "docling"
        except RuntimeError:
            pages = extract_pdf(path)
            extractor_used = "pymupdf"
        except Exception as exc:
            if "DocumentConversionError" in type(exc).__name__:
                pages = extract_pdf(path)
                extractor_used = "pymupdf"
            else:
                raise
    elif suffix == ".docx":
        from backend.app.ingestion.docling_extractor import extract_docling
        from backend.app.ingestion.docx_extractor import extract_docx
        try:
            pages = extract_docling(path)
            extractor_used = "docling"
        except Exception:
            pages = extract_docx(path)
            extractor_used = "docx"
    elif suffix == ".txt":
        from backend.app.ingestion.text_extractor import extract_text
        pages = extract_text(path)
        extractor_used = "text"
    else:
        raise ValueError(f"No text extractor registered for '{suffix}'")

    if not pages:
        return {"parents": [], "children": []}, "legalvault_e5", "e5_small", extractor_used

    # ── 2. Dispatcher picks embedding tier (based on file size/lang/pages) ────
    decision = dispatch(file_path=path, pages=pages)

    # ── 3. Hierarchical chunking: parents (SQLite only) + children (embedded)
    hier = chunk_hierarchical(pages)
    parents = hier["parents"]
    children = hier["children"]

    if not children:
        return {"parents": parents, "children": []}, decision.collection_name, decision.tier, extractor_used

    # ── 4. Embed only children ────────────────────────────────────────────────
    texts = [ch["content"] for ch in children]
    vectors = embed_batch(texts, model_name=decision.model_name)
    md = extra_metadata or {}
    payloads = [
        {
            "case_id": str(case_id),
            "doc_id": str(doc_id),
            "filename": filename,
            "content": ch["content"],
            "page": ch.get("page"),
            "section_title": ch.get("section_title"),
            "embedding_tier": decision.tier,
            "chunk_role": "child",
            "is_table": ch.get("is_table", False),
            "bbox": ch.get("bbox"),  # UI citation preview: TOPLEFT [x0,y0,x1,y1] or None
            **md,  # Phase 3: effective_date_ts, jurisdiction, version_tag, regulator
        }
        for ch in children
    ]
    qdrant_ids = insert(vectors, payloads, collection_name=decision.collection_name)
    for ch, qid in zip(children, qdrant_ids):
        ch["qdrant_id"] = qid

    return {"parents": parents, "children": children}, decision.collection_name, decision.tier, extractor_used


def _ingest_audio(
    path: str,
    doc_id: uuid.UUID,
    case_id: uuid.UUID,
    filename: str,
) -> tuple[list[dict], str, str, str]:
    """Transcribe + embed audio. Always uses e5-small-v2 (audio is always English-first).

    Returns (chunks, qdrant_collection, embedding_tier, extractor_used="whisper").
    """
    from backend.app.ingestion.audio_transcriber import transcribe_audio
    from backend.app.retrieval.embeddings import embed_batch
    from backend.app.retrieval.vector_store import COLLECTION_E5, insert

    segments = transcribe_audio(path)
    chunks = [
        {
            "content": s["text"],
            "ts_start": s.get("ts_start"),
            "ts_end": s.get("ts_end"),
            "page": None,
            "section_title": None,
        }
        for s in segments if s.get("text", "").strip()
    ]
    if not chunks:
        return [], COLLECTION_E5, "e5_small", "whisper"

    texts = [ch["content"] for ch in chunks]
    vectors = embed_batch(texts, model_name="intfloat/e5-small-v2")
    payloads = [
        {
            "case_id": str(case_id),
            "doc_id": str(doc_id),
            "filename": filename,
            "content": ch["content"],
            "ts_start": ch.get("ts_start"),
            "page": None,
            "section_title": None,
            "embedding_tier": "e5_small",
        }
        for ch in chunks
    ]
    qdrant_ids = insert(vectors, payloads, collection_name=COLLECTION_E5)
    for ch, qid in zip(chunks, qdrant_ids):
        ch["qdrant_id"] = qid

    return chunks, COLLECTION_E5, "e5_small", "whisper"


# ─── endpoints ───────────────────────────────────────────────────────────────


@router.post("/{case_id}/documents", summary="Upload and index a document")
async def upload_document(
    case_id: uuid.UUID,
    file: UploadFile,
    # ── Phase 3: optional metadata (all can be set later via PATCH) ───────────
    effective_date: Optional[str] = Form(default=None, description="ISO date (YYYY-MM-DD) when the doc took effect"),
    jurisdiction: Optional[str] = Form(default=None, description="e.g. 'India', 'Maharashtra', 'Delhi HC'"),
    version_tag: Optional[str] = Form(default=None, description="e.g. 'v2', '2026-01-15'"),
    regulator: Optional[str] = Form(default=None, description="e.g. 'RBI', 'SEBI', 'MCA'"),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(UserRole.analyst)),
):
    await get_case_for_user(case_id, session, user)
    effective_date_dt = _parse_effective_date(effective_date)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: PDF, TXT, DOCX, and audio files.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    doc_id = uuid.uuid4()
    storage_path = Path("./data/documents") / f"{doc_id}{suffix}"
    storage_path.write_bytes(content)

    doc = Document(
        id=doc_id,
        case_id=case_id,
        filename=file.filename or f"document{suffix}",
        doc_type=FileType.audio if suffix in _AUDIO_SUFFIXES else FileType.pdf,
        storage_path=str(storage_path),
        sha256=hashlib.sha256(content).hexdigest(),
        status=IngestStatus.indexing,
        effective_date=effective_date_dt,
        jurisdiction=jurisdiction,
        version_tag=version_tag,
        regulator=regulator,
    )
    session.add(doc)
    await session.commit()

    extra_metadata = _metadata_payload(doc)

    try:
        if suffix in _AUDIO_SUFFIXES:
            audio_chunks, qdrant_coll, emb_tier, extractor_used = await run_in_threadpool(
                _ingest_audio, str(storage_path), doc_id, case_id, doc.filename
            )
            # Audio path has no parent-child hierarchy; wrap flat chunks for the
            # persistence loop below.
            hier = {"parents": [], "children": audio_chunks}
            chunks = audio_chunks
        else:
            hier, qdrant_coll, emb_tier, extractor_used = await run_in_threadpool(
                _ingest_text_like, str(storage_path), doc_id, case_id, doc.filename, suffix, extra_metadata
            )
            chunks = hier["children"]
    except Exception as exc:
        doc.status = IngestStatus.error
        doc.error_message = str(exc)[:500]
        await session.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    # ── Persist PARENTS first so we can resolve child → parent FK ────────────
    parent_db_ids: list[uuid.UUID] = []
    for pi, p in enumerate(hier["parents"]):
        p_row = Chunk(
            document_id=doc_id,
            content=p["content"],
            chunk_index=pi,  # parent indices start at 0; children get their own below
            page_number=p.get("page"),
            section_title=p.get("section_title"),
            chunk_role="parent",
            parent_chunk_id=None,
            is_table=bool(p.get("is_table", False)),
        )
        session.add(p_row)
        parent_db_ids.append(p_row.id)
    if hier["parents"]:
        await session.flush()  # assign PKs so children can reference them

    for i, ch in enumerate(chunks):
        qdrant_uuid = None
        try:
            qdrant_uuid = uuid.UUID(ch["qdrant_id"]) if ch.get("qdrant_id") else None
        except ValueError:
            pass
        # Resolve child → parent FK via parent_index (set by hierarchical_chunker)
        p_idx = ch.get("parent_index")
        parent_fk = parent_db_ids[p_idx] if (p_idx is not None and 0 <= p_idx < len(parent_db_ids)) else None
        session.add(
            Chunk(
                document_id=doc_id,
                content=ch["content"],
                chunk_index=i,
                page_number=ch.get("page"),
                ts_start=ch.get("ts_start"),
                ts_end=ch.get("ts_end"),
                section_title=ch.get("section_title"),
                qdrant_id=qdrant_uuid,
                chunk_role="child",
                parent_chunk_id=parent_fk,
                is_table=bool(ch.get("is_table", False)),
            )
        )

    doc.status = IngestStatus.done
    doc.qdrant_collection = qdrant_coll
    doc.embedding_tier = emb_tier
    doc.extractor_used = extractor_used
    if suffix not in _AUDIO_SUFFIXES:
        doc.page_count = len({ch.get("page") for ch in chunks if ch.get("page")})

    await session.commit()
    return {
        "document_id": str(doc_id),
        "filename": doc.filename,
        "chunk_count": len(chunks),
        "status": "done",
        "embedding_tier": emb_tier,
        "qdrant_collection": qdrant_coll,
        "extractor_used": extractor_used,
    }


class DocumentMetadataUpdate(BaseModel):
    """All fields optional; only provided ones are updated."""
    effective_date: Optional[str] = None   # ISO date string
    jurisdiction: Optional[str] = None
    version_tag: Optional[str] = None
    regulator: Optional[str] = None


@router.patch(
    "/{case_id}/documents/{doc_id}/metadata",
    summary="Update document metadata (effective_date, jurisdiction, version_tag, regulator)",
)
async def update_document_metadata(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    body: DocumentMetadataUpdate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(UserRole.analyst)),
):
    """Patch metadata on a doc + reflect it in every chunk's Qdrant payload.

    Reindexing is lightweight (payload-only update, no re-embed) but requires
    rewriting all of this doc's points' payloads in the Qdrant collection.
    """
    await get_case_for_user(case_id, session, user)
    doc = await session.get(Document, doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Apply updates (only non-None fields)
    if body.effective_date is not None:
        doc.effective_date = _parse_effective_date(body.effective_date)
    if body.jurisdiction is not None:
        doc.jurisdiction = body.jurisdiction or None
    if body.version_tag is not None:
        doc.version_tag = body.version_tag or None
    if body.regulator is not None:
        doc.regulator = body.regulator or None

    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Rewrite Qdrant payloads for every child chunk of this doc so query-time
    # filters see the new values.
    from backend.app.retrieval.vector_store import set_payload_for_doc
    md = _metadata_payload(doc)
    try:
        updated = await run_in_threadpool(set_payload_for_doc, str(doc_id), doc.qdrant_collection, md)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qdrant payload update failed: {exc}") from exc

    return {
        "document_id": str(doc_id),
        "effective_date": doc.effective_date.isoformat() if doc.effective_date else None,
        "jurisdiction": doc.jurisdiction,
        "version_tag": doc.version_tag,
        "regulator": doc.regulator,
        "qdrant_payloads_updated": updated,
    }


@router.get("/{case_id}/documents/{doc_id}/status", summary="Poll ingestion status")
async def doc_status(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    await get_case_for_user(case_id, session, user)
    doc = await session.get(Document, doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks_result = await session.exec(select(Chunk).where(Chunk.document_id == doc_id))
    return {
        "document_id": str(doc_id),
        "status": doc.status,
        "chunk_count": len(chunks_result.all()),
        "embedding_tier": doc.embedding_tier,
        "qdrant_collection": doc.qdrant_collection,
        "extractor_used": doc.extractor_used,
        "error": doc.error_message,
    }


@router.delete(
    "/{case_id}/documents/{doc_id}", summary="Delete document and its vectors"
)
async def delete_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(UserRole.analyst)),
):
    await get_case_for_user(case_id, session, user)
    doc = await session.get(Document, doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found")

    from backend.app.retrieval.vector_store import delete_by_doc
    # Pass the specific collection so we don't scan both unnecessarily
    coll = getattr(doc, "qdrant_collection", None)
    removed = await run_in_threadpool(delete_by_doc, str(doc_id), coll)

    Path(doc.storage_path).unlink(missing_ok=True)
    await session.delete(doc)
    await session.commit()
    return {"deleted": True, "chunks_removed": removed}
