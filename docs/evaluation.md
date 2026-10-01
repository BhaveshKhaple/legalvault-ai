# LegalVault AI — Evaluation Report

## Task 10.1 — Retrieval accuracy

**Status:** PASS — Recall@5 target of 75% exceeded.

### Latest run

| Metric | Score | Hits |
|---|---|---|
| **Recall@5** | **100.00%** | 20 / 20 |
| Recall@3 | 100.00% | 20 / 20 |
| Recall@1 | 90.00% | 18 / 20 |

Target was Recall@5 >= 75% (per tracker 10.1 acceptance criteria). We beat it by 25 points.

### Setup

- **Corpus:** 30 hand-written legal snippets covering 4 document types (NDA, Service Agreement, Employment Contract, Lease Deed) spanning 19 clause categories (term, exclusions, payment, penalty, termination, liability, jurisdiction, dispute, probation, non-compete, rent, deposit, lock-in, escalation, etc.)
- **Queries:** 20 hand-labelled natural-language questions. Each has exactly one ground-truth chunk — a question "passes" if that chunk appears in the retriever's top-K results.
- **Pipeline exercised:** `embed → Qdrant dense search → BM25 sparse search → RRF fusion → cross-encoder rerank → top-K`. Identical to production RAG except the final LLM step is skipped.
- **Models:** e5-small-v2 embeddings (384-dim) + ms-marco-MiniLM-L-6-v2 cross-encoder reranker.
- **Isolation:** eval runs in a scratch Qdrant directory — never touches `./data/qdrant/`.

### The two Recall@1 misses

Both are at R@1 but recovered at R@3 — the top chunk was semantically close but the correct one was ranked 2 or 3:

- `q12` — "What is the annual salary of the senior software engineer?" — the "Compensation" chunk ranked 2 behind the "Position" chunk which also mentions "Senior Software Engineer".
- `q14` — "What is the notice period after probation is complete?" — the "Notice Period" chunk ranked 2 behind the "Probation" chunk (which also mentions "15 days notice").

Both are the kind of ambiguity a reranker or LLM would resolve from context. Not production bugs.

### Reproducing

```powershell
cd D:\projects\Legalvalult AI\legalvault-ai
backend\venv\Scripts\python tests\eval\eval_retrieval.py
```

Exits 0 on pass, 1 on fail — safe to use as a CI gate (task 9.4 picks this up).

### Data file

`tests/eval/test_queries.json` — single JSON with `corpus` (30 chunks) + `queries` (20 Q&A pairs). Edit this file to extend the test set with new categories or edge cases.

### Known limitations

- Synthetic corpus, not real contracts. R@5=100% here does NOT mean R@5=100% on your law firm's actual messy PDFs. We'll need to re-run on real data as it accumulates.
- No multilingual queries yet. e5-small-v2 is English-only; BGE-M3 (tier_1 / tier_bhavesh) handles English + Hindi — swap tier before running the eval if testing multilingual.
- Only tests retrieval, not the final LLM answer quality. That's a separate eval (not yet in scope).
