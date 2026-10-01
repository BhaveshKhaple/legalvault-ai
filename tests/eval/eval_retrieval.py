"""
Task 10.1 — Retrieval accuracy evaluation.

Loads tests/eval/test_queries.json (synthetic legal corpus + 20 hand-labelled
queries) and measures how often the correct chunk appears in the retriever's
top-K results.

Pipeline exercised (same as production RAG, minus the final LLM):
    query → embed → Qdrant dense search
          → BM25 sparse search
          → RRF fusion
          → cross-encoder rerank
          → top-K chunks

Metrics reported:
    - Recall@1 — correct chunk is #1
    - Recall@3 — correct chunk in top 3
    - Recall@5 — correct chunk in top 5  (tracker target: ≥ 0.75)
    - Per-category breakdown

Usage:
    cd legalvault-ai
    backend/venv/Scripts/python tests/eval/eval_retrieval.py

Uses a SCRATCH Qdrant directory so your real ./data/qdrant/ is not touched.
Exits with status 1 if Recall@5 falls below the 0.75 target (CI-friendly).
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

# Make sure we import from the repo root (works from any cwd)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Use a scratch Qdrant dir so this eval never touches ./data/qdrant/
_SCRATCH_QDRANT = Path(tempfile.mkdtemp(prefix="lv_eval_qdrant_"))
os.environ["QDRANT_PATH"] = str(_SCRATCH_QDRANT)

# After env is set, import the real pipeline
from backend.app.retrieval.bm25_index import build_bm25        # noqa: E402
from backend.app.retrieval.embeddings import embed, embed_batch  # noqa: E402
from backend.app.retrieval.hybrid import fuse                   # noqa: E402
from backend.app.retrieval.reranker import rerank               # noqa: E402
from backend.app.retrieval.vector_store import insert, query    # noqa: E402
from backend.app.llm.model_selector import get_model            # noqa: E402

TARGET_RECALL_AT_5 = 0.75


def _retrieve(question: str, case_id: str, corpus: list[dict], bm25_index, k: int = 5) -> list[dict]:
    """Run the full retrieval chain and return the top-k chunks."""
    tier = get_model()
    q_vec = embed(question, model_name=tier.embedding)

    dense = query(q_vec, case_id, top_k=20)
    sparse_raw = bm25_index.query(question, k=20)
    sparse = [{"chunk": r["chunk"], "score": r["score"]} for r in sparse_raw]

    fused = fuse(dense, sparse, k=20)
    if not fused:
        return []
    top = rerank(question, fused, k=k)
    return top


def _top_chunk_ids(top_chunks: list[dict]) -> list[str]:
    """Pull the stable chunk_id from each top result (from payload or chunk_id)."""
    out = []
    for c in top_chunks:
        # vector_store returns payload with original chunk_id
        payload = c.get("payload") or {}
        cid = payload.get("chunk_id") or c.get("chunk_id")
        if cid:
            out.append(str(cid))
    return out


def main() -> int:
    print("=" * 70)
    print("LegalVault AI - Task 10.1 retrieval evaluation")
    print("=" * 70)

    # ── 1. Load the dataset ──────────────────────────────────────────────────
    dataset_path = Path(__file__).parent / "test_queries.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    corpus = data["corpus"]
    queries = data["queries"]
    print(f"\nCorpus:  {len(corpus)} chunks")
    print(f"Queries: {len(queries)}  (one correct chunk expected per query)")

    tier = get_model()
    print(f"\nActive tier: {tier.tier}")
    print(f"  Embedding: {tier.embedding} ({tier.embedding_dim}-dim)")
    print(f"  LLM:       {tier.llm}  (not used in retrieval eval)")

    # ── 2. Build the scratch index ──────────────────────────────────────────
    case_id = f"eval-{uuid.uuid4()}"
    print(f"\nIndexing {len(corpus)} chunks into scratch Qdrant @ {_SCRATCH_QDRANT}")
    t0 = time.perf_counter()
    texts = [c["content"] for c in corpus]
    vectors = embed_batch(texts, model_name=tier.embedding)
    payloads = [
        {
            "case_id": case_id,
            "doc_id": c["doc_id"],
            "chunk_id": c["id"],  # stable id used for scoring
            "content": c["content"],
            "page": c.get("page"),
            "section_title": c.get("section"),
            "filename": c["doc_id"] + ".pdf",
        }
        for c in corpus
    ]
    insert(vectors, payloads)

    bm25_corpus = [
        {
            "chunk_id": c["id"],
            "content": c["content"],
            "page": c.get("page"),
            "doc_id": c["doc_id"],
            "section_title": c.get("section"),
        }
        for c in corpus
    ]
    bm25_index = build_bm25(bm25_corpus)
    print(f"Index built in {time.perf_counter() - t0:.1f}s")

    # ── 3. Run each query ────────────────────────────────────────────────────
    results = []
    per_cat_hits = defaultdict(lambda: {"n": 0, "r1": 0, "r3": 0, "r5": 0})
    totals = {"r1": 0, "r3": 0, "r5": 0}

    print("\n" + "-" * 70)
    print(f"{'ID':<5} {'Cat':<14} {'R@1':<4} {'R@5':<4}  Question")
    print("-" * 70)

    for q in queries:
        top = _retrieve(q["question"], case_id, corpus, bm25_index, k=5)
        ids = _top_chunk_ids(top)
        expected = q["expected_chunk_id"]

        r1 = expected in ids[:1]
        r3 = expected in ids[:3]
        r5 = expected in ids[:5]

        if r1: totals["r1"] += 1
        if r3: totals["r3"] += 1
        if r5: totals["r5"] += 1

        cat = q["category"]
        per_cat_hits[cat]["n"] += 1
        if r1: per_cat_hits[cat]["r1"] += 1
        if r3: per_cat_hits[cat]["r3"] += 1
        if r5: per_cat_hits[cat]["r5"] += 1

        results.append({
            **q,
            "retrieved_top5": ids,
            "r1": r1, "r3": r3, "r5": r5,
            "rank": (ids.index(expected) + 1) if expected in ids else None,
        })

        mark1 = "Y" if r1 else " "
        mark5 = "Y" if r5 else "X"
        print(f"{q['id']:<5} {cat:<14} {mark1:<4} {mark5:<4}  {q['question'][:50]}")

    # ── 4. Report ────────────────────────────────────────────────────────────
    n = len(queries)
    recall_1 = totals["r1"] / n
    recall_3 = totals["r3"] / n
    recall_5 = totals["r5"] / n

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"  Recall@1:  {recall_1:.2%}   ({totals['r1']}/{n})")
    print(f"  Recall@3:  {recall_3:.2%}   ({totals['r3']}/{n})")
    print(f"  Recall@5:  {recall_5:.2%}   ({totals['r5']}/{n})  target >={TARGET_RECALL_AT_5:.0%}")

    print("\nPER-CATEGORY RECALL@5")
    for cat in sorted(per_cat_hits):
        pc = per_cat_hits[cat]
        pct = pc["r5"] / pc["n"] if pc["n"] else 0
        print(f"  {cat:<14}  {pct:.2%}  ({pc['r5']}/{pc['n']})")

    # List misses — what the retriever got wrong
    misses = [r for r in results if not r["r5"]]
    if misses:
        print(f"\nMISSES ({len(misses)}):")
        for m in misses:
            print(f"  {m['id']}  expected={m['expected_chunk_id']}  got={m['retrieved_top5']}")
            print(f"         {m['question']}")

    # Clean up scratch Qdrant
    try:
        shutil.rmtree(_SCRATCH_QDRANT, ignore_errors=True)
    except Exception:
        pass

    # Exit code — pass/fail for CI gating
    if recall_5 >= TARGET_RECALL_AT_5:
        print(f"\nPASS  — Recall@5 {recall_5:.2%} meets {TARGET_RECALL_AT_5:.0%} target.\n")
        return 0
    print(f"\nFAIL  — Recall@5 {recall_5:.2%} below {TARGET_RECALL_AT_5:.0%} target.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
