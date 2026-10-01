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
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import get_case_for_user, get_current_user, require_role
from backend.app.database import get_session
from backend.app.models import Case, Chunk, Document, FileType, IngestStatus, User, UserRole

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
) -> tuple[list[dict], str, str]:
    """Extract, chunk, dispatch, embed, index a text-format document.

    Returns:
        (chunks, qdrant_collection, embedding_tier)
    """
    from backend.app.ingestion.clause_chunker import chunk_legal_doc
    from backend.app.ingestion.embedding_dispatcher import dispatch
    from backend.app.retrieval.embeddings import embed_batch
    from backend.app.retrieval.vector_store import insert

    if suffix == ".pdf":
        from backend.app.ingestion.pdf_extractor import extract_pdf
        pages = extract_pdf(path)
    elif suffix == ".txt":
        from backend.app.ingestion.text_extractor import extract_text
        pages = extract_text(path)
    elif suffix == ".docx":
        from backend.app.ingestion.docx_extractor import extract_docx
        pages = extract_docx(path)
    else:
        raise ValueError(f"No text extractor registered for '{suffix}'")

    if not pages:
        return [], "legalvault_e5", "e5_small"

    # ── embedding dispatcher ──────────────────────────────────────────────────
    decision = dispatch(file_path=path, pages=pages)

    chunks = chunk_legal_doc(pages)
    if not chunks:
        return [], decision.collection_name, decision.tier

    texts = [ch["content"] for ch in chunks]
    vectors = embed_batch(texts, model_name=decision.model_name)
    payloads = [
        {
            "case_id": str(case_id),
            "doc_id": str(doc_id),
            "filename": filename,
            "content": ch["content"],
            "page": ch.get("page"),
            "section_title": ch.get("section_title"),
            "embedding_tier": decision.tier,
        }
        for ch in chunks
    ]
    qdrant_ids = insert(vectors, payloads, collection_name=decision.collection_name)
    for ch, qid in zip(chunks, qdrant_ids):
        ch["qdrant_id"] = qid

    return chunks, decision.collection_name, decision.tier


def _ingest_audio(
    path: str,
    doc_id: uuid.UUID,
    case_id: uuid.UUID,
    filename: str,
) -> tuple[list[dict], str, str]:
    """Transcribe + embed audio. Always uses e5-small-v2 (audio is always English-first)."""
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
        return [], COLLECTION_E5, "e5_small"

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

    return chunks, COLLECTION_E5, "e5_small"


# ─── endpoints ───────────────────────────────────────────────────────────────


@router.post("/{case_id}/documents", summary="Upload and index a document")
async def upload_document(
    case_id: uuid.UUID,
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(UserRole.analyst)),
):
    case = await get_case_for_user(case_id, session, user)

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
    )
    session.add(doc)
    await session.commit()

    try:
        if suffix in _AUDIO_SUFFIXES:
            chunks, qdrant_coll, emb_tier = await run_in_threadpool(
                _ingest_audio, str(storage_path), doc_id, case_id, doc.filename
            )
        else:
            chunks, qdrant_coll, emb_tier = await run_in_threadpool(
                _ingest_text_like, str(storage_path), doc_id, case_id, doc.filename, suffix
            )
    except Exception as exc:
        doc.status = IngestStatus.error
        doc.error_message = str(exc)[:500]
        await session.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    for i, ch in enumerate(chunks):
        qdrant_uuid = None
        try:
            qdrant_uuid = uuid.UUID(ch["qdrant_id"]) if ch.get("qdrant_id") else None
        except ValueError:
            pass
        session.add(Chunk(
            document_id=doc_id,
            content=ch["content"],
            chunk_index=i,
            page_number=ch.get("page"),
            ts_start=ch.get("ts_start"),
            ts_end=ch.get("ts_end"),
            section_title=ch.get("section_title"),
            qdrant_id=qdrant_uuid,
        ))

    doc.status = IngestStatus.done
    doc.qdrant_collection = qdrant_coll
    doc.embedding_tier = emb_tier
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
        "error": doc.error_message,
    }


@router.delete("/{case_id}/documents/{doc_id}", summary="Delete document and its vectors")
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
