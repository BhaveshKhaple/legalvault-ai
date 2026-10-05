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
    build_metadata_filter,
    query as qdrant_query,
    query_all_collections,
    query_with_filter,
)

# Phase 3: widen first-stage recall. The cross-encoder narrows to 5 after,
# so this doesn't hurt LLM latency; it does catch previously-missed chunks
# that the dense+BM25 top-20 was dropping just outside the cutoff.
FIRST_STAGE_K = 100

_MODEL_FOR_COLLECTION = {
    COLLECTION_E5: "intfloat/e5-small-v2",
    COLLECTION_BGE: "BAAI/bge-m3",
}


async def run_rag(
    case_id: uuid.UUID,
    question: str,
    session: AsyncSession,
    metadata_filters: dict | None = None,
) -> dict:
    """Execute the full RAG pipeline for a question against a case's documents.

    Returns:
        dict with keys: answer, evidence, confidence, latency_ms
    """
    t0 = time.perf_counter()

    # ── 1. Fetch CHILD chunks for this case (BM25 corpus) ───────────────────
    # Phase 2: parents live in SQLite but are never indexed. BM25 and Qdrant
    # both see only children; parents are swapped in at the LLM-context stage.
    # Legacy rows (no chunk_role set) default to "child" via the model default,
    # so pre-Phase-2 cases keep working untouched.
    chunks_result = await session.exec(
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.case_id == case_id)
        .where(Chunk.chunk_role != "parent")
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

    # Phase 3: if metadata filters were sent, build a combined Qdrant Filter
    # and use the filter-aware query path. Otherwise keep the fast legacy path.
    has_filters = bool(metadata_filters)
    if len(vectors_by_collection) == 1:
        coll, vec = next(iter(vectors_by_collection.items()))
        if has_filters:
            qf = build_metadata_filter(str(case_id), **metadata_filters)
            dense_results = query_with_filter(vec, qf, top_k=FIRST_STAGE_K, collection_name=coll)
        else:
            dense_results = qdrant_query(vec, str(case_id), top_k=FIRST_STAGE_K, collection_name=coll)
    else:
        # Mixed-tier case — fan-out, results merged by score before RRF.
        # (Phase 3 filter applies to each collection in the fan-out.)
        if has_filters:
            all_results = []
            for coll, vec in vectors_by_collection.items():
                qf = build_metadata_filter(str(case_id), **metadata_filters)
                all_results.extend(query_with_filter(vec, qf, top_k=FIRST_STAGE_K, collection_name=coll))
            all_results.sort(key=lambda r: r["score"], reverse=True)
            dense_results = all_results
        else:
            dense_results = query_all_collections(vectors_by_collection, str(case_id), top_k=FIRST_STAGE_K)

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
    # Phase 2 — Parent swap:
    #   Retrieved top-K chunks are CHILDREN (~200 chars, precise).
    #   For the LLM we fetch each child's PARENT (~1500 chars, full section)
    #   and dedupe parents when multiple children point to the same one.
    #   Evidence shown to the UI stays at child-level for citation precision.
    from backend.app.llm.prompts.citation_prompt import EvidenceChunk, build_citation_prompt
    from backend.app.llm.ollama_client import generate
    from backend.app.llm.model_selector import get_model

    tier = get_model()
    _MAX_CONTENT = 1800   # bigger budget per chunk now — parents are longer

    # Resolve parent_chunk_id for each retrieved child, then fetch parent texts.
    # chunk_id in top_chunks is the qdrant point id (string UUID). We look up
    # the matching Chunk row to get its parent_chunk_id.
    retrieved_qdrant_ids: list[str] = []
    for c in top_chunks:
        cid = c.get("chunk_id") or (c.get("payload") or {}).get("chunk_id")
        if cid:
            retrieved_qdrant_ids.append(str(cid))

    # Build child-row map keyed by qdrant_id string, then collect parent IDs.
    child_rows_by_qid: dict[str, Chunk] = {}
    if retrieved_qdrant_ids:
        import uuid as _uuid
        qdrant_uuids = []
        for s in retrieved_qdrant_ids:
            try:
                qdrant_uuids.append(_uuid.UUID(s))
            except ValueError:
                pass
        if qdrant_uuids:
            rows = (await session.exec(
                select(Chunk).where(Chunk.qdrant_id.in_(qdrant_uuids))
            )).all()
            for r in rows:
                child_rows_by_qid[str(r.qdrant_id)] = r

    # Fetch parents in bulk — dedupe keeps the LLM context focused.
    parent_ids = []
    seen_parents = set()
    for c in top_chunks:
        cid = str(c.get("chunk_id") or (c.get("payload") or {}).get("chunk_id") or "")
        row = child_rows_by_qid.get(cid)
        if row and row.parent_chunk_id and row.parent_chunk_id not in seen_parents:
            parent_ids.append(row.parent_chunk_id)
            seen_parents.add(row.parent_chunk_id)
    parents_by_id: dict = {}
    if parent_ids:
        parent_rows = (await session.exec(
            select(Chunk).where(Chunk.id.in_(parent_ids))
        )).all()
        parents_by_id = {p.id: p for p in parent_rows}

    evidence_chunks: list[EvidenceChunk] = []
    for c in top_chunks:
        payload = c.get("payload", {})
        cid = str(c.get("chunk_id") or payload.get("chunk_id") or "")
        child_row = child_rows_by_qid.get(cid)

        # LLM context: prefer parent text if available; else child text.
        if child_row and child_row.parent_chunk_id and child_row.parent_chunk_id in parents_by_id:
            llm_text = parents_by_id[child_row.parent_chunk_id].content
        else:
            llm_text = payload.get("content") or c.get("content", "")
        llm_text = llm_text[:_MAX_CONTENT] + ("…" if len(llm_text) > _MAX_CONTENT else "")

        evidence_chunks.append(EvidenceChunk(
            chunk_id=cid,
            filename=payload.get("filename", "document"),
            content=llm_text,
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
            # UI citation preview: doc_id lets the frontend build the
            # /v1/cases/{cid}/documents/{did}/page-image?page=N URL; bbox is
            # the TOPLEFT [x0,y0,x1,y1] rectangle to overlay on that rendered
            # page (None when the extractor was PyMuPDF/text/whisper).
            "doc_id": c.get("payload", {}).get("doc_id"),
            "bbox": c.get("payload", {}).get("bbox"),
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
