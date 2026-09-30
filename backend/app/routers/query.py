"""Task 6.1 stub — query endpoint. Full RAG pipeline in Task 6.3."""

import uuid
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/{case_id}/query", summary="Semantic query (stub — Task 6.3)")
async def run_query(case_id: uuid.UUID, body: QueryRequest):
    return {
        "answer": "stub — full RAG pipeline wired in Task 6.3",
        "evidence": [],
        "confidence": 0.0,
        "latency_ms": 0,
    }
