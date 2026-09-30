"""Task 6.2 — Cases CRUD endpoints (auth wired in Task 6.4)."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.database import get_session
from backend.app.models import Case, DocType, Document

router = APIRouter()


class CaseCreate(BaseModel):
    name: str
    client_name: Optional[str] = None
    doc_type: DocType = DocType.contract


@router.post("", summary="Create a new case")
async def create_case(body: CaseCreate, session: AsyncSession = Depends(get_session)):
    case = Case(name=body.name, client_name=body.client_name, doc_type=body.doc_type)
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return {"case_id": str(case.id), "name": case.name, "doc_type": case.doc_type}


@router.get("", summary="List all cases")
async def list_cases(session: AsyncSession = Depends(get_session)):
    result = await session.exec(select(Case))
    cases = result.all()
    out = []
    for c in cases:
        docs_result = await session.exec(select(Document).where(Document.case_id == c.id))
        out.append({
            "case_id": str(c.id),
            "name": c.name,
            "client_name": c.client_name,
            "doc_type": c.doc_type,
            "document_count": len(docs_result.all()),
            "created_at": c.created_at.isoformat(),
        })
    return out


@router.get("/{case_id}", summary="Get case with document list")
async def get_case(case_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    docs_result = await session.exec(select(Document).where(Document.case_id == case_id))
    doc_list = [
        {
            "document_id": str(d.id),
            "filename": d.filename,
            "doc_type": d.doc_type,
            "status": d.status,
            "page_count": d.page_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs_result.all()
    ]
    return {
        "case_id": str(case.id),
        "name": case.name,
        "client_name": case.client_name,
        "doc_type": case.doc_type,
        "created_at": case.created_at.isoformat(),
        "documents": doc_list,
    }


@router.delete("/{case_id}", summary="Delete a case and all its documents/chunks")
async def delete_case(case_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    from backend.app.models import Chunk
    from backend.app.retrieval.vector_store import delete_by_doc
    from fastapi.concurrency import run_in_threadpool

    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load documents first, then delete chunks and Qdrant vectors per doc
    docs_result = await session.exec(select(Document).where(Document.case_id == case_id))
    docs = docs_result.all()
    total_chunks = 0
    for doc in docs:
        # Remove Qdrant vectors
        removed = await run_in_threadpool(delete_by_doc, str(doc.id))
        total_chunks += removed
        # Remove chunks from SQLite
        chunks_result = await session.exec(select(Chunk).where(Chunk.document_id == doc.id))
        for ch in chunks_result.all():
            await session.delete(ch)
        # Remove file from disk
        try:
            from pathlib import Path as _Path
            _Path(doc.storage_path).unlink(missing_ok=True)
        except Exception:
            pass
        await session.delete(doc)

    await session.delete(case)
    await session.commit()
    return {"deleted": True, "case_id": str(case_id), "docs_removed": len(docs), "chunks_removed": total_chunks}
