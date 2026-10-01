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

---

## eval-corpus — mixed-format end-to-end evaluation

**Status (retrieval-only):** PASS — Recall@5 95.7% on 23 queries across 10 documents.
**Status (--with-llm):** PASS — LLM answer hit rate 73.9%, refusal 100%, all 4 gates satisfied.

### What this adds

The 10.1 retrieval eval used a hand-written synthetic chunk corpus loaded as JSON. This new eval runs the FULL ingestion pipeline against real documents on disk in 3 different formats (PDF, TXT, DOCX) with 8 different structural patterns, so we're exercising extractors + chunker + embedding + hybrid retrieval + optional LLM — not just the retrieval sub-chain.

### Corpus (10 synthetic documents)

Generated by `tests/eval/build_corpus.py` into `tests/eval/corpus/`. Each doc targets a different real-world structure:

| ID | File | Format | Structure | Pages |
|---|---|---|---|---|
| d01 | d01_nda_numbered.pdf | PDF | numbered-clauses (1.1, 1.2) | 2 |
| d02 | d02_service_agreement.pdf | PDF | long-numbered-sections | 3 |
| d03 | d03_employment_contract.pdf | PDF | prose + bullet lists | 2 |
| d04 | d04_lease_deed.pdf | PDF | defined-terms-heavy | 3 |
| d05 | d05_privacy_policy.pdf | PDF | heading-driven, no numbering | 3 |
| d06 | d06_fee_schedule.pdf | PDF | table-heavy quotation | 1 |
| d07 | d07_meeting_minutes.txt | TXT | unstructured prose | 1 |
| d08 | d08_product_spec.docx | DOCX | H1/H2/H3 headings | 10 |
| d09 | d09_faq.docx | DOCX | Q&A format | 1 |
| d10 | d10_bilingual_notice.pdf | PDF | English + Hindi legal notice | 1 |

Hand-labelled queries (23) + out-of-corpus adversarial queries (3) live in `tests/eval/corpus_manifest.json`.

### Gates

| Gate | Target | Blocking? |
|---|---|---|
| Chunker — every doc produces >=1 chunk (no empty extractions) | 100% | yes |
| Retrieval Recall@5 (overall) | >=80% | yes |
| LLM answer hit rate (`--with-llm`) | >=70% | yes when flag set |
| Refusal rate on adversarial queries (`--with-llm`) | >=66% | yes when flag set |

### Latest retrieval-only run

```
Recall@1  82.6%  (19/23)
Recall@3  87.0%  (20/23)
Recall@5  95.7%  (22/23)
```

Per-doc R@5: d01-d06, d08-d10 all 100% (2-3 queries each). Only d07 at 50% — the second meeting-minutes query asks "why" (requires chaining two sentences), which single-chunk retrieval doesn't nail.

### Latest --with-llm run

```
LLM answer hit rate  73.9%  (17/23)       target 70%  PASS
Refusal (adversarial)  100%  (3/3)        target 66%  PASS
```

Per-doc LLM accuracy — retrieval was 100% on all except d07, so these differences are purely about LLM extraction quality from the retrieved chunks:

| Doc | Structure | LLM hit |
|---|---|---|
| d01 | numbered-clauses | 100% |
| d02 | long-numbered-sections | 100% |
| d03 | prose + bullets (1-chunk) | 33% |
| d04 | defined-terms-heavy | 100% |
| d05 | heading-driven (1-chunk) | 0% |
| d06 | table-heavy | 100% |
| d07 | unstructured prose (txt) | 0% |
| d08 | H1/H2/H3 headings (docx) | 100% |
| d09 | Q&A (docx, 1-chunk) | 100% |
| d10 | bilingual English+Hindi | 100% |

**Correlation worth noting:** every doc where the chunker produced one giant 1500-1800 char chunk either scored 100% (when the answer is a verbatim phrase, like d09's "No. All processing runs on your own hardware") or scored very poorly (d03 33%, d05 0%). phi3:mini can echo a verbatim string from a big chunk, but it struggles to extract a specific number or date when the chunk is long and the surrounding context has distractors. This is a known small-model limitation, not a pipeline bug.

Refusal was 100% on all 3 adversarial queries (France capital, Bitcoin price, maternity policy). The citation prompt's "I do not have evidence" instruction fires correctly when retrieved chunks are irrelevant.

### Finding — chunker collapses non-numbered documents

Four multi-page docs collapsed to 1 chunk despite having multiple pages:

| ID | Pages | Chunks | Chars/chunk | Structure |
|---|---|---|---|---|
| d03 | 2 | 1 | 1365 | prose + bullets |
| d05 | 3 | 1 | 1761 | heading-driven |
| d06 | 1 | 1 | 822 | table-heavy |
| d09 | 1 | 1 | 1807 | Q&A |

**Why:** `backend/app/ingestion/clause_chunker.py` (Task 1.2) only splits on numbered-clause boundaries (`1.1`, `2.3`, `Section 4`). Documents written as flowing prose, Q&A, or tables have no such boundaries, so they come out as one chunk per logical page (and when a doc has no form-feed markers, that collapses to one chunk per document).

**Consequence:** Retrieval still works (BM25 + semantic search both tolerate large chunks fine — all these docs score 100% R@5 above). But:
- Citation granularity is coarse (chunk = whole doc, not clause).
- LLM context is wider per chunk (1500-1800 chars), closer to the 2000-char default context, so less room for other evidence.
- The reranker has less to rank — 1 candidate instead of many.

**Fix (future work):** add a fallback chunker that splits on heading styles for DOCX, on paragraph breaks + character count for prose PDFs, and on form-feed / long-char-count for TXT. Not in scope for this branch — the primary quality signal (retrieval + LLM answer) is still green.

### New extractors added by this branch

- `backend/app/ingestion/text_extractor.py` — `.txt` support with form-feed-aware virtual pagination
- `backend/app/ingestion/docx_extractor.py` — `.docx` with heading-driven virtual pagination + table flattening
- `backend/app/routers/documents.py` — extension-based extractor dispatch; `_ALLOWED_SUFFIXES` extended to `.pdf | .txt | .docx | <audio>`

New deps pinned in `backend/requirements.txt`: `python-docx>=1.1`, `reportlab>=4.2` (reportlab is dev-only for `build_corpus.py`).

### Reproducing

```powershell
# 1. Rebuild the 10-doc corpus (idempotent)
backend\venv\Scripts\python tests\eval\build_corpus.py

# 2. Fast — retrieval only, no Ollama needed
backend\venv\Scripts\python tests\eval\eval_corpus.py

# 3. Full — retrieval + LLM answers + refusal (needs Ollama up on :11434)
backend\venv\Scripts\python tests\eval\eval_corpus.py --with-llm
```

### Known limitations of this eval

- All 10 docs are synthetic. Real contracts have scanned pages, inconsistent fonts, and OCR errors; this corpus has none of those.
- Only one tier tested per run (whatever `MODEL_TIER` is in `.env`). The embedding dispatcher (next branch) will let the eval exercise both e5-small-v2 and BGE-M3 simultaneously.
- Refusal detection uses substring matching on known refusal phrases. A model that refuses in a novel way would be scored as a false fail.

