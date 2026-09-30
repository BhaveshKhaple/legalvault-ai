"""Task 6.3 — Query endpoint: wires the full RAG pipeline."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.database import get_session
from backend.app.models import Case
from backend.app.services.rag_service import run_rag

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/{case_id}/query", summary="Ask a question about a case's documents")
async def run_query(
    case_id: uuid.UUID,
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    case = await session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")

    # RAG pipeline includes CPU-bound embedding + Ollama HTTP call
    result = await run_rag(case_id, body.question, session)
    return result
