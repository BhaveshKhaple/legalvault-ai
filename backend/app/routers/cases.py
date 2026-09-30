"""Task 6.1 stub — cases routes. Full CRUD in Task 6.2+."""

import uuid
from fastapi import APIRouter

router = APIRouter()


@router.post("", summary="Create case (stub — full impl Task 6.2)")
async def create_case():
    return {"case_id": str(uuid.uuid4()), "detail": "stub"}


@router.get("", summary="List cases (stub)")
async def list_cases():
    return []


@router.get("/{case_id}", summary="Get case detail (stub)")
async def get_case(case_id: uuid.UUID):
    return {"id": str(case_id), "detail": "stub"}
