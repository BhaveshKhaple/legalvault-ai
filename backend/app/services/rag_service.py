"""
Task 6.3 — RAG pipeline service.
embedding-dispatcher — Step 3 now fans out to both Qdrant collections when a
case contains docs indexed by different embedding models. Each collection gets
a correctly-dimensioned query vector; results are merged before RRF fusion.

Full chain:
  question → embed (per collection) → BM25 + Qdrant fan-out hybrid → rerank
           → LLM → answer
"""

import math
import time
import uuid

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.models import Chunk, Document
from backend.app.retrieval.vector_store import (
    COLLECTION_BGE, COLLECTION_E5,
    query as qdrant_query,
    query_all_collections,
)

_MODEL_FOR_COLLECTION = {
    COLLECTION_E5: "intfloat/e5-small-v2",
    COLLECTION_BGE: "BAAI/bge-m3",
}


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
    sparse_results = [{"chunk": r["chunk"], "score": r["score"]} for r in bm25_raw]

    # ── 3. Semantic (dense) search — fan-out to all collections in this case ──
    # Discover which Qdrant collections hold this case's documents.
    doc_ids = list({ch.document_id for ch in db_chunks})
    docs_result = await session.exec(
        select(Document).where(Document.id.in_(doc_ids))
    )
    docs = docs_result.all()
    collections_in_case = {
        getattr(d, "qdrant_collection", COLLECTION_E5) or COLLECTION_E5
        for d in docs
    }

    from backend.app.retrieval.embeddings import embed

    # Build a correctly-dimensioned query vector per collection
    vectors_by_collection = {
        coll: embed(question, model_name=_MODEL_FOR_COLLECTION.get(coll, "intfloat/e5-small-v2"))
        for coll in collections_in_case
    }

    if len(vectors_by_collection) == 1:
        coll, vec = next(iter(vectors_by_collection.items()))
        dense_results = qdrant_query(vec, str(case_id), top_k=20, collection_name=coll)
    else:
        # Mixed-tier case — fan-out, results merged by score before RRF
        dense_results = query_all_collections(vectors_by_collection, str(case_id), top_k=20)

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
    from backend.app.llm.model_selector import get_model

    tier = get_model()
    _MAX_CONTENT = 600
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
    def _norm(x):
        return 1.0 / (1.0 + math.exp(-float(x)))

    evidence_out = [
        {
            "rank": i + 1,
            "score": _norm(c.get("rerank_score", c.get("score", 0))),
            "content": ev.content,
            "filename": ev.filename,
            "page": ev.page,
            "ts_start": ev.ts_start,
            "section_title": c.get("payload", {}).get("section_title"),
            "embedding_tier": c.get("payload", {}).get("embedding_tier", "e5_small"),
        }
        for i, (c, ev) in enumerate(zip(top_chunks, evidence_chunks))
    ]

    raw = float(top_chunks[0].get("rerank_score", top_chunks[0].get("score", 0)))
    confidence = 1.0 / (1.0 + math.exp(-raw))

    return {
        "answer": answer,
        "evidence": evidence_out,
        "confidence": round(confidence, 4),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
