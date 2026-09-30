"""
Task 6.2 — Document upload endpoint with synchronous ingestion pipeline.

Upload flow (synchronous for local dev — ARQ queue added in Task 9.3):
  1. Receive PDF or audio via multipart POST
  2. Save to data/documents/<doc_id>.<ext>
  3. Extract text (pdf_extractor) or transcribe (whisper)
  4. Chunk into clauses (clause_chunker)
  5. Embed chunks using the active tier's embedding model
  6. Insert vectors + payloads to Qdrant (case_id for tenant isolation)
  7. Persist Chunk records to SQLite (BM25 corpus at query time)
  8. Update Document status → done
"""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.database import get_session
from backend.app.models import Case, Chunk, Document, FileType, IngestStatus

router = APIRouter()

_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_ALLOWED_SUFFIXES = {".pdf"} | _AUDIO_SUFFIXES


# ─── sync ingestion helpers (run in thread pool) ─────────────────────────────


def _ingest_pdf(path: str, doc_id: uuid.UUID, case_id: uuid.UUID, filename: str) -> list[dict]:
    from backend.app.ingestion.pdf_extractor import extract_pdf
    from backend.app.ingestion.clause_chunker import chunk_legal_doc
    from backend.app.retrieval.embeddings import embed_batch
    from backend.app.retrieval.vector_store import insert
    from backend.app.llm.model_selector import get_model

    pages = extract_pdf(path)
    if not pages:
        return []
    chunks = chunk_legal_doc(pages)
    if not chunks:
        return []

    texts = [ch["content"] for ch in chunks]
    vectors = embed_batch(texts, model_name=get_model().embedding)
    payloads = [
        {
            "case_id": str(case_id),
            "doc_id": str(doc_id),
            "filename": filename,
            "content": ch["content"],
            "page": ch.get("page"),
            "section_title": ch.get("section_title"),
        }
        for ch in chunks
    ]
    qdrant_ids = insert(vectors, payloads)
    for ch, qid in zip(chunks, qdrant_ids):
        ch["qdrant_id"] = qid
    return chunks


def _ingest_audio(path: str, doc_id: uuid.UUID, case_id: uuid.UUID, filename: str) -> list[dict]:
    from backend.app.ingestion.audio_transcriber import transcribe_audio
    from backend.app.retrieval.embeddings import embed_batch
    from backend.app.retrieval.vector_store import insert
    from backend.app.llm.model_selector import get_model

    segments = transcribe_audio(path)
    chunks = [
        {"content": s["text"], "ts_start": s.get("ts_start"), "ts_end": s.get("ts_end"),
         "page": None, "section_title": None}
        for s in segments if s.get("text", "").strip()
    ]
    if not chunks:
        return []

    texts = [ch["content"] for ch in chunks]
    vectors = embed_batch(texts, model_name=get_model().embedding)
    payloads = [
        {
            "case_id": str(case_id),
            "doc_id": str(doc_id),
            "filename": filename,
            "content": ch["content"],
            "ts_start": ch.get("ts_start"),
            "page": None,
            "section_title": None,
        }
        for ch in chunks
    ]
    qdrant_ids = insert(vectors, payloads)
    for ch, qid in zip(chunks, qdrant_ids):
        ch["qdrant_id"] = qid
    return chunks


# ─── endpoints ───────────────────────────────────────────────────────────────


@router.post("/{case_id}/documents", summary="Upload and index a document")
async def upload_document(
    case_id: uuid.UUID,
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
):
    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: PDF and audio files.",
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

    # Run CPU-bound ingestion in thread pool — keeps event loop free
    try:
        ingest_fn = _ingest_audio if suffix in _AUDIO_SUFFIXES else _ingest_pdf
        chunks = await run_in_threadpool(
            ingest_fn, str(storage_path), doc_id, case_id, doc.filename
        )
    except Exception as exc:
        doc.status = IngestStatus.error
        doc.error_message = str(exc)[:500]
        await session.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    # Persist chunk records for BM25 rebuild at query time
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
    if suffix != ".pdf":
        pass
    else:
        doc.page_count = len({ch.get("page") for ch in chunks if ch.get("page")})

    await session.commit()
    return {"document_id": str(doc_id), "filename": doc.filename,
            "chunk_count": len(chunks), "status": "done"}


@router.get("/{case_id}/documents/{doc_id}/status", summary="Poll ingestion status")
async def doc_status(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    doc = await session.get(Document, doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks_result = await session.exec(select(Chunk).where(Chunk.document_id == doc_id))
    return {
        "document_id": str(doc_id),
        "status": doc.status,
        "chunk_count": len(chunks_result.all()),
        "error": doc.error_message,
    }


@router.delete("/{case_id}/documents/{doc_id}", summary="Delete document and its vectors")
async def delete_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    doc = await session.get(Document, doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found")

    from backend.app.retrieval.vector_store import delete_by_doc
    removed = await run_in_threadpool(delete_by_doc, str(doc_id))

    Path(doc.storage_path).unlink(missing_ok=True)
    await session.delete(doc)
    await session.commit()
    return {"deleted": True, "chunks_removed": removed}
