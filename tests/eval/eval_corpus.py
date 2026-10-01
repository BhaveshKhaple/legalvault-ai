"""
eval-corpus — End-to-end evaluation across the 10-document mixed-format
test corpus.

Measures four orthogonal quality signals against the SAME production
pipeline the FastAPI backend uses:

  1. Chunker health     — chunks produced per doc, avg length, no empties
  2. Retrieval          — Recall@1 / Recall@3 / Recall@5 on manifest queries
                          (match = top-K chunk contains the required substring)
  3. LLM answer quality — [opt-in --with-llm] does the generated answer
                          contain at least one expected substring?
  4. Refusal behaviour  — [opt-in --with-llm] on adversarial out-of-corpus
                          questions, does the LLM say "I do not have evidence"?

Scratch Qdrant — never touches ./data/qdrant/.

Usage:
    # Retrieval only (fast — no Ollama needed)
    backend/venv/Scripts/python tests/eval/eval_corpus.py

    # Full pipeline including LLM answers + refusal (slow, needs Ollama)
    backend/venv/Scripts/python tests/eval/eval_corpus.py --with-llm

Exit codes:
    0 — all gates pass
    1 — one or more gates fail
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Scratch Qdrant before importing anything that opens it
_SCRATCH_QDRANT = Path(tempfile.mkdtemp(prefix="lv_eval_corpus_"))
os.environ["QDRANT_PATH"] = str(_SCRATCH_QDRANT)

from backend.app.ingestion.clause_chunker import chunk_legal_doc        # noqa: E402
from backend.app.ingestion.docx_extractor import extract_docx           # noqa: E402
from backend.app.ingestion.pdf_extractor import extract_pdf             # noqa: E402
from backend.app.ingestion.text_extractor import extract_text           # noqa: E402
from backend.app.llm.model_selector import get_model                    # noqa: E402
from backend.app.retrieval.bm25_index import build_bm25                 # noqa: E402
from backend.app.retrieval.embeddings import embed, embed_batch         # noqa: E402
from backend.app.retrieval.hybrid import fuse                           # noqa: E402
from backend.app.retrieval.reranker import rerank                       # noqa: E402
from backend.app.retrieval.vector_store import insert, query            # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
MANIFEST = Path(__file__).parent / "corpus_manifest.json"

# Gate thresholds — tune in one place
GATE_RETRIEVAL_R5 = 0.80       # overall Recall@5 target
# The clause_chunker (task 1.2) is designed to split on numbered-clause
# boundaries. Documents without that structure (prose, Q&A, tables) are
# expected to come out as a single large chunk. Gating only on 0 chunks.
GATE_CHUNKER_MIN_PER_DOC = 1
GATE_LLM_HIT_RATE = 0.70       # when --with-llm, this fraction of answers must match
GATE_REFUSAL_RATE = 0.66       # when --with-llm, this fraction of adversarial queries must refuse


EXTRACTORS = {
    "pdf": extract_pdf,
    "txt": extract_text,
    "docx": extract_docx,
}


def _extract(path: Path, fmt: str) -> list[dict]:
    try:
        return EXTRACTORS[fmt](str(path))
    except KeyError:
        raise ValueError(f"No extractor for format '{fmt}'")


def _retrieve(question: str, case_id: str, bm25_index, k: int = 5) -> list[dict]:
    tier = get_model()
    q_vec = embed(question, model_name=tier.embedding)
    dense = query(q_vec, case_id, top_k=20)
    sparse_raw = bm25_index.query(question, k=20)
    sparse = [{"chunk": r["chunk"], "score": r["score"]} for r in sparse_raw]
    fused = fuse(dense, sparse, k=20)
    if not fused:
        return []
    return rerank(question, fused, k=k)


def _chunk_contains(chunk: dict, needle: str) -> bool:
    """Case-insensitive substring match against a retrieved chunk."""
    payload = chunk.get("payload") or {}
    text = payload.get("content") or chunk.get("content") or chunk.get("chunk", {}).get("content") or ""
    return needle.lower() in text.lower()


def _hit_at_k(top: list[dict], needle: str, k: int) -> bool:
    for ch in top[:k]:
        if _chunk_contains(ch, needle):
            return True
    return False


def _generate_answer(question: str, case_id: str, bm25_index) -> str:
    """Full RAG with LLM — mirrors backend/app/services/rag_service.run_rag."""
    import uuid as _uuid
    from backend.app.llm.ollama_client import generate
    from backend.app.llm.prompts.citation_prompt import EvidenceChunk, build_citation_prompt

    tier = get_model()
    top = _retrieve(question, case_id, bm25_index, k=5)
    if not top:
        return "I do not have evidence in the uploaded documents to answer this question."

    evidence_chunks = []
    for c in top:
        payload = c.get("payload") or {}
        evidence_chunks.append(EvidenceChunk(
            chunk_id=payload.get("chunk_id") or str(_uuid.uuid4()),
            filename=payload.get("filename") or "doc",
            page=payload.get("page"),
            ts_start=payload.get("ts_start"),
            content=(payload.get("content") or "")[:600],
        ))

    prompt = build_citation_prompt(question, evidence_chunks)
    try:
        return generate(prompt, model=tier.llm)
    except Exception as exc:
        return f"[LLM-ERROR] {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="also measure LLM answer quality + refusal (needs Ollama)")
    args = ap.parse_args()

    print("=" * 72)
    print("LegalVault AI - eval-corpus (10-doc mixed-format evaluation)")
    print("=" * 72)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = manifest["documents"]
    adversarial = manifest.get("adversarial_queries", [])

    tier = get_model()
    print(f"\nActive tier:     {tier.tier}")
    print(f"  Embedding:     {tier.embedding} ({tier.embedding_dim}-dim)")
    print(f"  LLM:           {tier.llm} {'(exercised)' if args.with_llm else '(skipped)'}")
    print(f"\nDocuments:       {len(docs)}")
    total_queries = sum(len(d['queries']) for d in docs)
    print(f"Queries:         {total_queries}")
    print(f"Adversarial:     {len(adversarial)}")
    print(f"Scratch Qdrant:  {_SCRATCH_QDRANT}")

    # ─── 1. Ingest every document ──────────────────────────────────────────
    case_id = f"eval-corpus-{uuid.uuid4()}"
    all_bm25_chunks: list[dict] = []
    chunker_stats: list[dict] = []

    print("\n" + "-" * 72)
    print(f"{'ID':<5} {'Fmt':<5} {'Pages':<6} {'Chunks':<7} {'AvgLen':<7} Structure")
    print("-" * 72)

    for d in docs:
        path = CORPUS_DIR / d["filename"]
        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        pages = _extract(path, d["format"])
        chunks = chunk_legal_doc(pages) if pages else []
        n_chunks = len(chunks)
        avg_len = int(sum(len(c["content"]) for c in chunks) / n_chunks) if n_chunks else 0

        chunker_stats.append({
            "id": d["id"],
            "format": d["format"],
            "pages": len(pages),
            "chunks": n_chunks,
            "avg_chunk_len": avg_len,
            "structure": d["structure"],
        })

        if n_chunks == 0:
            print(f"{d['id']:<5} {d['format']:<5} {len(pages):<6} 0       -       {d['structure']} [EMPTY]")
            continue

        texts = [c["content"] for c in chunks]
        vectors = embed_batch(texts, model_name=tier.embedding)
        payloads = [
            {
                "case_id": case_id,
                "doc_id": d["id"],
                "chunk_id": f"{d['id']}_c{i}",
                "content": c["content"],
                "page": c.get("page"),
                "filename": d["filename"],
                "section_title": c.get("section_title"),
            }
            for i, c in enumerate(chunks)
        ]
        insert(vectors, payloads)

        for i, c in enumerate(chunks):
            all_bm25_chunks.append({
                "chunk_id": f"{d['id']}_c{i}",
                "content": c["content"],
                "page": c.get("page"),
                "doc_id": d["id"],
                "section_title": c.get("section_title"),
            })

        print(f"{d['id']:<5} {d['format']:<5} {len(pages):<6} {n_chunks:<7} {avg_len:<7} {d['structure']}")

    bm25_index = build_bm25(all_bm25_chunks)

    # ─── 2. Run queries ────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("RETRIEVAL + ANSWER EVAL")
    print("-" * 72)
    header = f"{'ID':<10} {'R@1':<4} {'R@3':<4} {'R@5':<4}"
    if args.with_llm:
        header += "  LLM  "
    header += " Question"
    print(header)

    retr_hits = {"r1": 0, "r3": 0, "r5": 0}
    retr_total = 0
    llm_hits = 0
    llm_total = 0

    per_doc = {}
    for d in docs:
        per_doc[d["id"]] = {"n": 0, "r5": 0, "llm": 0}
        for q in d["queries"]:
            retr_total += 1
            per_doc[d["id"]]["n"] += 1
            top = _retrieve(q["question"], case_id, bm25_index, k=5)
            needle = q["must_contain_in_chunk"]
            h1 = _hit_at_k(top, needle, 1)
            h3 = _hit_at_k(top, needle, 3)
            h5 = _hit_at_k(top, needle, 5)
            if h1: retr_hits["r1"] += 1
            if h3: retr_hits["r3"] += 1
            if h5: retr_hits["r5"] += 1
            if h5: per_doc[d["id"]]["r5"] += 1

            line = f"{q['id']:<10} {'Y' if h1 else ' ':<4} {'Y' if h3 else ' ':<4} {'Y' if h5 else 'X':<4}"

            if args.with_llm:
                llm_total += 1
                answer = _generate_answer(q["question"], case_id, bm25_index)
                lower = answer.lower()
                any_match = any(sub.lower() in lower for sub in q["must_contain_in_answer"])
                if any_match:
                    llm_hits += 1
                    per_doc[d["id"]]["llm"] += 1
                line += f"  {'Y' if any_match else 'X'}    "

            line += f" {q['question'][:48]}"
            print(line)

    # ─── 3. Adversarial / refusal ─────────────────────────────────────────
    refusal_hits = 0
    if args.with_llm and adversarial:
        print("\n" + "-" * 72)
        print("REFUSAL BEHAVIOUR (out-of-corpus questions)")
        print("-" * 72)
        for aq in adversarial:
            answer = _generate_answer(aq["question"], case_id, bm25_index)
            lower = answer.lower()
            refused = any(phrase in lower for phrase in [
                "do not have evidence",
                "cannot answer",
                "insufficient evidence",
                "no evidence",
                "not covered",
                "not available in the provided",
                "not in the documents",
            ])
            if refused:
                refusal_hits += 1
            mark = "Y" if refused else "X"
            print(f"{aq['id']:<8} {mark}  {aq['question']}")
            if not refused:
                print(f"           got: {answer[:120]}...")

    # ─── 4. Verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GATES")
    print("=" * 72)

    # Chunker gate
    bad_chunker_docs = [c for c in chunker_stats if c["chunks"] < GATE_CHUNKER_MIN_PER_DOC]
    chunker_pass = not bad_chunker_docs
    print(f"[{'PASS' if chunker_pass else 'FAIL'}] chunker — every doc produced >= {GATE_CHUNKER_MIN_PER_DOC} chunks")
    if bad_chunker_docs:
        for b in bad_chunker_docs:
            print(f"       {b['id']} ({b['format']}, {b['structure']}): {b['chunks']} chunks")

    # Retrieval gate
    r1 = retr_hits["r1"] / retr_total if retr_total else 0
    r3 = retr_hits["r3"] / retr_total if retr_total else 0
    r5 = retr_hits["r5"] / retr_total if retr_total else 0
    retr_pass = r5 >= GATE_RETRIEVAL_R5
    print(f"[{'PASS' if retr_pass else 'FAIL'}] retrieval — Recall@5 {r5:.1%} (target >= {GATE_RETRIEVAL_R5:.0%})")
    print(f"       Recall@1 {r1:.1%}  Recall@3 {r3:.1%}  Recall@5 {r5:.1%}   ({retr_hits['r5']}/{retr_total})")

    # LLM gate
    llm_pass = True
    if args.with_llm:
        llm_rate = llm_hits / llm_total if llm_total else 0
        llm_pass = llm_rate >= GATE_LLM_HIT_RATE
        print(f"[{'PASS' if llm_pass else 'FAIL'}] LLM answer — hit rate {llm_rate:.1%} (target >= {GATE_LLM_HIT_RATE:.0%})   ({llm_hits}/{llm_total})")

    # Refusal gate
    refusal_pass = True
    if args.with_llm and adversarial:
        rr = refusal_hits / len(adversarial)
        refusal_pass = rr >= GATE_REFUSAL_RATE
        print(f"[{'PASS' if refusal_pass else 'FAIL'}] refusal   — rate {rr:.1%} (target >= {GATE_REFUSAL_RATE:.0%})   ({refusal_hits}/{len(adversarial)})")

    # Per-doc summary
    print("\nPER-DOC RETRIEVAL@5")
    for doc_id, stat in per_doc.items():
        pct = stat["r5"] / stat["n"] if stat["n"] else 0
        extra = ""
        if args.with_llm:
            lpct = stat["llm"] / stat["n"] if stat["n"] else 0
            extra = f"  LLM {lpct:.0%}"
        print(f"  {doc_id}  {pct:.0%}  ({stat['r5']}/{stat['n']}){extra}")

    all_pass = chunker_pass and retr_pass and llm_pass and refusal_pass

    try:
        shutil.rmtree(_SCRATCH_QDRANT, ignore_errors=True)
    except Exception:
        pass

    print("\n" + ("PASS - all gates satisfied" if all_pass else "FAIL - one or more gates below target"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
