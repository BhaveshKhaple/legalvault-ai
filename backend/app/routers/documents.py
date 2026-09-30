"""Task 6.1 stub — document upload routes. Full impl in Task 6.2."""

import uuid
from fastapi import APIRouter

router = APIRouter()


@router.post("/{case_id}/documents", summary="Upload document (stub — Task 6.2)")
async def upload_document(case_id: uuid.UUID):
    return {"document_id": str(uuid.uuid4()), "task_id": "stub", "detail": "Task 6.2"}


@router.get("/{case_id}/documents/{doc_id}/status", summary="Poll ingestion status (stub)")
async def doc_status(case_id: uuid.UUID, doc_id: uuid.UUID):
    return {"status": "done", "chunk_count": 0}


@router.delete("/{case_id}/documents/{doc_id}", summary="Delete document (stub)")
async def delete_document(case_id: uuid.UUID, doc_id: uuid.UUID):
    return {"deleted": True, "chunks_removed": 0}
