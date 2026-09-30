"""Task 6.1 stub — Report Shield endpoint. Full impl in Task 5.x + 6.x."""

import uuid
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ShieldRequest(BaseModel):
    source_doc_id: uuid.UUID


@router.post("/{case_id}/shield", summary="Report Shield audit (stub — Tasks 5.x)")
async def run_shield(case_id: uuid.UUID, body: ShieldRequest):
    return {
        "trust_score": 0.0,
        "verified_count": 0,
        "unverified_count": 0,
        "contradiction_count": 0,
        "missing_sections": [],
        "flags": [],
        "detail": "stub — full pipeline in Tasks 5.1-5.4",
    }
