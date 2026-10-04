"""
Task 5.3/5.4 — Report Shield endpoint.

Full pipeline per uploaded report document:
  1. Fetch doc text from disk (already extracted and stored at upload time)
  2. Extract claims (claim_extractor)
  3. Detect missing required sections (gap_detector, doc_type from Document record)
  4. Verify each claim against the case's indexed docs (verifier, Reverse-RAG)
  5. Aggregate into a trust score (scoring)
  6. Return full Shield result
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.auth.dependencies import get_case_for_user, require_role
from backend.app.database import get_session
from backend.app.models import Document, ShieldReport, User, UserRole

router = APIRouter()


class ShieldRequest(BaseModel):
    source_doc_id: uuid.UUID


@router.post("/{case_id}/shield", summary="Run Report Shield audit on a document")
async def run_shield(
    case_id: uuid.UUID,
    body: ShieldRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(UserRole.analyst)),
):
    await get_case_for_user(case_id, session, user)

    doc = await session.get(Document, body.source_doc_id)
    if not doc or doc.case_id != case_id:
        raise HTTPException(status_code=404, detail="Document not found in this case")

    if doc.doc_type.value == "audio":
        raise HTTPException(
            status_code=422,
            detail="Report Shield requires a PDF document, not audio",
        )

    result = await run_in_threadpool(_run_shield_pipeline, doc, str(case_id))

    # Persist shield report
    report = ShieldReport(
        case_id=case_id,
        source_doc_id=body.source_doc_id,
        trust_score=float(result["trust_score"]),
        verified_count=result["verified_count"],
        unverified_count=result["unverified_count"],
        contradiction_count=result["contradiction_count"],
        missing_sections=json.dumps(result["missing_sections"]),
        flags=json.dumps(result["contradictions"]),
    )
    session.add(report)
    await session.commit()

    return result


def _run_shield_pipeline(doc: Document, case_id: str) -> dict:
    """Synchronous shield pipeline — run in thread pool."""
    import logging
    from backend.app.ingestion.pdf_extractor import extract_pdf
    from backend.app.ingestion.docling_extractor import extract_docling
    from backend.app.shield.claim_extractor import extract_claims
    from backend.app.shield.gap_detector import detect_gaps_from_pages
    from backend.app.shield.verifier import verify_claims
    from backend.app.shield.scoring import aggregate

    logger = logging.getLogger(__name__)

    # 1. Extract text from stored PDF (PyMuPDF is instant ~40ms; Docling OCR as fallback)
    try:
        pages = extract_pdf(doc.storage_path)
        if not pages or not any(p.get("text", "").strip() for p in pages):
            logger.info(
                f"PyMuPDF found no text in '{doc.filename}', trying Docling OCR fallback"
            )
            pages = extract_docling(doc.storage_path)
    except Exception as exc:
        logger.warning(
            f"PyMuPDF extraction failed for '{doc.filename}', trying Docling: {exc}"
        )
        try:
            pages = extract_docling(doc.storage_path)
        except Exception:
            pages = []

    if not pages:
        return {
            "trust_score": 0,
            "band": "at_risk",
            "band_label": "At risk — do not file",
            "verified_count": 0,
            "unverified_count": 0,
            "contradiction_count": 0,
            "missing_count": 0,
            "contradictions": [],
            "missing_sections": [],
            "flags": [],
            "detail": "Could not extract text from document (possibly scanned PDF)",
        }

    # 2. Extract claims
    full_text = "\n".join(p.get("text", "") for p in pages)
    extracted = extract_claims(full_text)
    # Filter trivial/fragment items and keep substantive claims (max 25)
    substantive = [c for c in extracted if len(c.strip()) >= 15]
    claims = substantive[:25] if len(substantive) > 25 else (substantive or extracted)

    # 3. Detect missing sections
    doc_type_str = (
        doc.doc_type.value if hasattr(doc.doc_type, "value") else str(doc.doc_type)
    )
    # Map FileType → template doc_type
    template_type = "contract"  # default
    if "audit" in doc_type_str:
        template_type = "audit_report"
    elif "agreement" in doc_type_str:
        template_type = "agreement"

    gap_report = detect_gaps_from_pages(pages, template_type)

    # 4. Verify claims (Reverse-RAG)
    # Discover which Qdrant collections hold this case's docs so the verifier
    # can fan-out correctly (e.g. legalvault_e5 AND/OR legalvault_bge).
    import sqlite3 as _sqlite3
    import os as _os

    _db_url = _os.environ.get(
        "DATABASE_URL", "sqlite+aiosqlite:///./data/legalvault.db"
    )
    _db_path = _db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    try:
        _conn = _sqlite3.connect(_db_path)
        _cur = _conn.cursor()
        _cur.execute(
            "SELECT DISTINCT qdrant_collection FROM documents "
            "WHERE case_id=? AND status='done' AND qdrant_collection IS NOT NULL",
            (case_id,),
        )
        _collections = [r[0] for r in _cur.fetchall()] or ["legalvault_e5"]
        _conn.close()
    except Exception:
        _collections = ["legalvault_e5"]

    verification_results = verify_claims(claims, case_id, collections=_collections)

    # 5. Aggregate trust score
    score_result = aggregate(verification_results, gap_report.to_dict())

    return {
        **score_result.to_dict(),
        "total_claims": len(claims),
        "flags": score_result.contradictions,
        "verified_claims": [
            r for r in verification_results if r.get("status") == "Verified"
        ],
        "unverified_claims": [
            r for r in verification_results if r.get("status") == "Unverified"
        ],
    }
