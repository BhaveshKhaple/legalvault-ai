"""Task 6.3 — Query endpoint: wires the full RAG pipeline."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
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

    try:
        result = await run_rag(case_id, body.question, session)
        return result
    except Exception as exc:
        # Surface actionable errors to the UI instead of a bare 500
        msg = str(exc)
        if "OllamaUnavailable" in type(exc).__name__ or "11434" in msg or "Connection" in msg:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve",
            )
        if "model" in msg.lower() and ("not found" in msg.lower() or "pull" in msg.lower()):
            from backend.app.llm.model_selector import get_model
            tier = get_model()
            raise HTTPException(
                status_code=503,
                detail=f"LLM model '{tier.llm}' not found in Ollama. Run: ollama pull {tier.llm}",
            )
        raise HTTPException(status_code=500, detail=f"Query failed: {msg[:300]}")
