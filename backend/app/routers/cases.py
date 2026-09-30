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


@router.delete("/{case_id}", summary="Delete a case")
async def delete_case(case_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    await session.delete(case)
    await session.commit()
    return {"deleted": True, "case_id": str(case_id)}
