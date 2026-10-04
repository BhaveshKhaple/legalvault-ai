"""
Task 6.3 — Query endpoint: wires the full RAG pipeline.
Task 6.4 — Auth + ownership check via get_case_for_user.
Phase 3  — Optional metadata filter params (date / jurisdiction / version / regulator).
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import get_case_for_user, get_current_user
from backend.app.database import get_session
from backend.app.models import User
from backend.app.services.rag_service import run_rag

router = APIRouter()


class QueryFilters(BaseModel):
    """Phase 3 — optional metadata filters applied server-side at retrieval time."""
    date_after: Optional[str] = Field(default=None, description="ISO date; keep docs effective on/after this date")
    date_before: Optional[str] = Field(default=None, description="ISO date; keep docs effective on/before this date")
    jurisdiction: Optional[str] = Field(default=None, description="exact match on Document.jurisdiction")
    version_tag: Optional[str] = Field(default=None, description="exact match on Document.version_tag")
    regulator: Optional[str] = Field(default=None, description="exact match on Document.regulator")


class QueryRequest(BaseModel):
    question: str
    filters: Optional[QueryFilters] = None


def _iso_to_ts(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail=f"Invalid date filter: '{raw}'")


@router.post("/{case_id}/query", summary="Ask a question about a case's documents")
async def run_query(
    case_id: uuid.UUID,
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    await get_case_for_user(case_id, session, user)

    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")

    # Convert filter dates to unix ts so Qdrant Range comparator can match the
    # effective_date_ts int stamped on every chunk payload.
    filter_kwargs: dict = {}
    if body.filters:
        filter_kwargs = {
            "date_after": _iso_to_ts(body.filters.date_after),
            "date_before": _iso_to_ts(body.filters.date_before),
            "jurisdiction": body.filters.jurisdiction or None,
            "version_tag": body.filters.version_tag or None,
            "regulator": body.filters.regulator or None,
        }
        # Drop keys that are all-None so rag_service can treat the dict as a presence flag.
        filter_kwargs = {k: v for k, v in filter_kwargs.items() if v is not None}

    try:
        result = await run_rag(case_id, body.question, session, metadata_filters=filter_kwargs or None)
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
