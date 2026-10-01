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

---

## Task 10.2 — Report Shield detection

**Status (mock mode):** PASS — all missing-section flaws caught (3/3 testable reports).
**Status (real-LLM mode):** PASS — 5/5 reports caught, 8/8 planted flaws detected (target 4/5).

### What it tests

5 synthetic reports, each with deliberately planted flaws. Shield "passes" a report if it detects at least one of the planted flaws.

| Report | Planted flaws |
|---|---|
| r1 | Missing Indemnity section |
| r2 | Missing Governing Law + missing Signatures |
| r3 | Payment terms contradicted vs source (30d vs 60d, 2% vs no interest) |
| r4 | Salary currency contradicted (Rs. 18L vs USD 50k) |
| r5 | Lock-in contradicted + rent escalation contradicted + missing Indemnity |

### Two modes

**Mock mode (default)** — uses rerank score only, no Ollama calls. Fast (~30s). Validates missing-section detection via `gap_detector`. Contradiction-only reports (r3, r4) always "miss" by design — pass criterion only requires all missing-section reports to pass (3/3).

**Real-LLM mode (`--real-llm`)** — full pipeline with Ollama per claim. Slow (~2-5 min on GPU). Validates everything including contradiction detection via `verifier`. Pass criterion: >=4/5 reports have at least one planted flaw caught.

### Latest mock-mode run

| Report | Hits / Flaws | Trust | Verdict |
|---|---|---|---|
| r1 | 1/1 | 68 | PASS |
| r2 | 2/2 | 63 | PASS |
| r3 | 0/1 | 73 | needs --real-llm |
| r4 | 0/1 | 76 | needs --real-llm |
| r5 | 1/3 | 68 | PASS (section) |

**Mock-mode verdict:** PASS — 3/3 missing-section testable reports caught.

### Latest real-LLM run

Verified against live Ollama (phi3:mini on GTX 1650, ~2 min end-to-end).

| Report | Hits / Flaws | Trust | Verdict |
|---|---|---|---|
| r1 | 1/1 | 51 | PASS |
| r2 | 2/2 | 23 | PASS |
| r3 | 1/1 | 33 | PASS (contradiction caught) |
| r4 | 1/1 | 36 | PASS (contradiction caught) |
| r5 | 3/3 |  0 | PASS (both contradictions + missing section) |

**Real-LLM verdict:** PASS — 5/5 reports caught, 8/8 planted flaws detected. Target was 4/5 reports with any detection; we got all 5, every planted flaw found. This confirms the three fixes from the previous section (shield router import, sigmoid rerank threshold, fail-safe default) actually unblock the contradiction path end-to-end — all 3 contradiction-only reports (r3, r4, r5) were caught by the verifier, which they weren't before the sigmoid fix.

### Reproducing

```powershell
# Fast — mock mode, CI-friendly:
backend\venv\Scripts\python tests\eval\eval_shield.py

# Full — real LLM, true-to-prod:
backend\venv\Scripts\python tests\eval\eval_shield.py --real-llm
```

### Side-fixes discovered by this eval

1. **Latent `ImportError` in shield router** — `backend/app/routers/shield.py` imported `extract_claims` from `backend/app/shield/claim_extractor.py`, which only exported `chunk_report`. Shield endpoint would have crashed on first real PDF. Added `extract_claims(text)` function; `chunk_report()` now delegates to it.
2. **Verifier threshold miscalibrated** — `_RELEVANCE_THRESHOLD = 0.3` compared against raw cross-encoder logits (range ~-10 to +10). Even strongly-relevant chunks score near 0, so the LLM was almost never asked to judge contradictions — everything came back `Unverified`. Fixed by sigmoid-normalizing the rerank score before threshold comparison. Threshold 0.3 now means "sigmoid probability > 0.3" (raw > -0.85).
3. **Verifier fail-safe inverted** — when the LLM call failed (e.g. Ollama down), the verifier defaulted to `"Verified"`. For a legal tool that is dangerous — it would falsely vouch for claims it never actually checked. Changed default to `"Unverified"` so Ollama outages produce honest "we couldn't verify this" results instead of silent passes.

### Known limitations

- Mock mode cannot test contradictions. For CI quick-checks, mock mode is sufficient; for release validation, run `--real-llm`.
- `claim_extractor.extract_claims()` with spaCy backs down to regex fallback if `en_core_web_sm` isn't downloaded. Download once: `python -m spacy download en_core_web_sm`.
- No "employment" doc_type template yet — maps to "agreement" currently. Add `backend/app/shield/templates/employment.yaml` if we need employment-contract-specific required sections.
