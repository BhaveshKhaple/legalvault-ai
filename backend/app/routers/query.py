"""
Task 6.3 — Query endpoint: wires the full RAG pipeline.
Task 6.4 — Auth + ownership check via get_case_for_user.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import get_case_for_user, get_current_user
from backend.app.database import get_session
from backend.app.models import User
from backend.app.services.rag_service import run_rag

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/{case_id}/query", summary="Ask a question about a case's documents")
async def run_query(
    case_id: uuid.UUID,
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    case = await get_case_for_user(case_id, session, user)

    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")

    try:
        result = await run_rag(case_id, body.question, session)
        return result
    except Exception as exc:
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
