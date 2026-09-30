"""
Task 6.3 — RAG pipeline service.

Full chain: question → embed → BM25 + Qdrant hybrid → rerank → LLM → answer.

BM25 corpus is rebuilt from the Chunk records in SQLite on every query.
This is correct for a single-user local install; a production service would
cache the BM25 index per case and invalidate on new uploads.
"""

import time
import uuid
from typing import Any

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.models import Chunk, Document


async def run_rag(case_id: uuid.UUID, question: str, session: AsyncSession) -> dict:
    """Execute the full RAG pipeline for a question against a case's documents.

    Returns:
        dict with keys: answer, evidence, confidence, latency_ms
    """
    t0 = time.perf_counter()

    # ── 1. Fetch all chunks for this case (BM25 corpus) ─────────────────────
    chunks_result = await session.exec(
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.case_id == case_id)
    )
    db_chunks: list[Chunk] = chunks_result.all()

    if not db_chunks:
        return {
            "answer": "No documents have been indexed for this case. Please upload a document first.",
            "evidence": [],
            "confidence": 0.0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # Build corpus list with qdrant_id as chunk_id for cross-path dedup
    corpus = [
        {
            "chunk_id": str(ch.qdrant_id) if ch.qdrant_id else str(ch.id),
            "content": ch.content,
            "page": ch.page_number,
            "section_title": ch.section_title,
            "ts_start": ch.ts_start,
            "doc_id": str(ch.document_id),
        }
        for ch in db_chunks
    ]

    # ── 2. BM25 sparse search ────────────────────────────────────────────────
    from backend.app.retrieval.bm25_index import build_bm25

    bm25_index = build_bm25(corpus)
    bm25_raw = bm25_index.query(question, k=20)
    # Normalise to the format fuse() expects for sparse results
    sparse_results = [
        {"chunk": r["chunk"], "score": r["score"]}
        for r in bm25_raw
    ]

    # ── 3. Semantic (dense) search via Qdrant ────────────────────────────────
    from backend.app.retrieval.embeddings import embed
    from backend.app.retrieval.vector_store import query as qdrant_query
    from backend.app.llm.model_selector import get_model

    tier = get_model()
    q_vec = embed(question, model_name=tier.embedding)
    dense_results = qdrant_query(q_vec, str(case_id), top_k=20)

    # ── 4. Hybrid fusion (RRF) ───────────────────────────────────────────────
    from backend.app.retrieval.hybrid import fuse

    fused = fuse(dense_results, sparse_results, k=20)

    if not fused:
        return {
            "answer": "No relevant excerpts found for this question.",
            "evidence": [],
            "confidence": 0.0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ── 5. Cross-encoder rerank ──────────────────────────────────────────────
    from backend.app.retrieval.reranker import rerank

    top_chunks = rerank(question, fused, k=5)

    # ── 6. Build citation prompt ─────────────────────────────────────────────
    from backend.app.llm.prompts.citation_prompt import EvidenceChunk, build_citation_prompt
    from backend.app.llm.ollama_client import generate

    _MAX_CONTENT = 600  # chars per chunk — keeps total prompt under ~2k tokens
    evidence_chunks: list[EvidenceChunk] = []
    for c in top_chunks:
        payload = c.get("payload", {})
        content = payload.get("content") or c.get("content", "")
        content = content[:_MAX_CONTENT] + ("…" if len(content) > _MAX_CONTENT else "")
        evidence_chunks.append(EvidenceChunk(
            chunk_id=c.get("chunk_id", ""),
            filename=payload.get("filename", "document"),
            content=content,
            page=payload.get("page"),
            ts_start=payload.get("ts_start"),
        ))

    prompt = build_citation_prompt(question, evidence_chunks)
    answer = generate(prompt, model=tier.llm)

    # ── 7. Format response ───────────────────────────────────────────────────
    evidence_out = [
        {
            "rank": i + 1,
            "score": float(c.get("rerank_score", c.get("score", 0))),
            "content": ev.content,
            "filename": ev.filename,
            "page": ev.page,
            "ts_start": ev.ts_start,
            "section_title": c.get("payload", {}).get("section_title"),
        }
        for i, (c, ev) in enumerate(zip(top_chunks, evidence_chunks))
    ]

    confidence = float(top_chunks[0].get("rerank_score", top_chunks[0].get("score", 0)))

    return {
        "answer": answer,
        "evidence": evidence_out,
        "confidence": round(min(confidence, 1.0), 4),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
