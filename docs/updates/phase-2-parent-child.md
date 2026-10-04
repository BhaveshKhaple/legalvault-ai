# Phase 2 — Parent-child chunking

## What I built

- **`backend/app/ingestion/hierarchical_chunker.py`** — produces two kinds of chunks per document:
  - **child** (~200 chars): one per clause / paragraph / table row. Embedded, stored in Qdrant, used for retrieval.
  - **parent** (~1500 chars): the full section containing a child. Stored only in SQLite; swapped in at LLM-context time.
- **`backend/app/models.py` — `Chunk` gets three new columns**: `chunk_role` (child / parent), `parent_chunk_id` (FK to chunks.id), `is_table` (bool). Legacy rows default to `chunk_role="child"` so Phase 1 cases keep working untouched. SQLite auto-migrate (`_sync_create_and_migrate`) picks up all three columns on next startup — no manual Alembic step needed.
- **`backend/app/routers/documents.py`** — upload path now calls `chunk_hierarchical()`, persists parents first (gets UUIDs via `session.flush()`), then persists children with resolved `parent_chunk_id` FK. Only children get embedded + indexed to Qdrant.
- **`backend/app/services/rag_service.py`** — after rerank returns top-5 children, the service fetches each child's parent via `parent_chunk_id`, dedupes parents when 2+ children share one, and sends the parents (not the children) to the LLM. Evidence citations shown to the UI stay at child-level for precision.
- **Short-item merge** — consecutive body items under 180 chars (CHILD_MERGE_BELOW_CHARS) are concatenated into one child. Fixes the Q/A failure mode where "Q: ..." and "A: ..." land as separate retrieval units and BM25 ranks the question over the answer.
- **Docling table export** — `docling_extractor.py` now calls `item.export_to_markdown()` when a `table` item has empty `.text`. Fixes the "fee schedule" failure where numeric cells were silently dropped.

## Files touched

- `backend/app/ingestion/hierarchical_chunker.py` (new, ~290 lines)
- `backend/app/ingestion/docling_extractor.py` (table markdown export)
- `backend/app/models.py` (Chunk + 3 columns)
- `backend/app/routers/documents.py` (chunker swap + parent persistence)
- `backend/app/services/rag_service.py` (child→parent swap at prompt stage)
- `tests/ingestion/test_hierarchical_chunker.py` (new, 10 tests)
- `tests/eval/eval_corpus.py` (hierarchical pipeline + parent-aware LLM eval)
- `scripts/smoke_hierarchical.py` + `scripts/debug_chunker.py` (dev-only)

## Eval results

| Gate | Pre-Phase-2 | Phase 2 | Target | Verdict |
|---|---|---|---|---|
| Chunker — >=1 chunk per doc | PASS | PASS | PASS | ✅ |
| Retrieval Recall@5 | 95.7% | **91.3%** | ≥80% | ✅ (4.4pt dip inside gate) |
| LLM answer hit rate | 73.9% | **91.3%** | ≥70% (ours ≥85%) | ✅ **+17.4 points** |
| Refusal on adversarial queries | 100% | 100% | ≥66% | ✅ |

Per-doc LLM scores — 9 of 10 docs at 100%, one at 0%:

| Doc | Pre-Phase-2 | Phase 2 | Note |
|---|---|---|---|
| d01 NDA numbered-clause | 100% | 0% | regression — parent swap may be giving phi3:mini too much context on this specific doc |
| d02 long service agreement | 100% | 100% | |
| d03 employment contract (prose+bullets) | 33% | 100% | **+67** — parent swap fix |
| d04 lease deed defined-terms | 100% | 100% | |
| d05 privacy policy heading-driven | 0% | 100% | **+100** — Docling hierarchy unlock |
| d06 fee schedule table-heavy | 100% | 100% | table export fix held |
| d07 meeting minutes (txt) | 0% | 100% | **+100** — hierarchical fallback helped |
| d08 product spec DOCX | 100% | 100% | |
| d09 FAQ DOCX Q&A | 100% | 100% | Q+A merge preserved semantics |
| d10 bilingual English+Hindi | 100% | 100% | |

The d01 regression is 2 queries out of 23 — net is still +17.4pt. Worth investigating but doesn't block Phase 2.

## How to test

```powershell
# unit
backend\venv\Scripts\python -m pytest tests\ingestion\test_hierarchical_chunker.py -q

# eval — retrieval only (fast)
backend\venv\Scripts\python tests\eval\eval_corpus.py

# eval — full pipeline including LLM (slow, ~5 min with Ollama up)
backend\venv\Scripts\python tests\eval\eval_corpus.py --with-llm
```

## Known gaps

- d01 (numbered-clause NDA) regressed from 100% → 0% LLM accuracy. Retrieval still correct; something about the parent swap confuses phi3:mini on this doc. Follow-up branch should check whether sending the parent instead of the child adds a distracting sibling clause for d01's two questions, and consider a per-doc toggle.
- The `_emit_children_for_docling` second-pass uses substring matching to assign children to parents. If a doc has two paragraphs with identical text, both land under the first parent. Not seen in the corpus but worth a positional index fix before real client data.
- `docx_extractor.py` is now dead code for the upload path (Docling handles DOCX directly). Left in-place as safety fallback; could be removed in a cleanup PR.
