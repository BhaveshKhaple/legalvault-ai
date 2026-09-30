"""Task 6.1 stub — auth routes. Full implementation in Task 6.4."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/login", summary="Password login (stub — Task 6.4)")
async def login():
    return {"detail": "Auth not yet implemented — see Task 6.4"}
