# LegalVault AI — Architecture

**Version:** 1.0 · **Owner:** Bhavesh Khaple (Team Pied Piper) · **Last updated:** 23 Sep 2026

Source of truth for the 5-layer architecture. Mirrors the diagram in TRD.md §2 and Slide 4 of the pitch deck.

---

## 1. Five layers

```
Ingestion  →  Processing  →  Retrieval  →  Audit  →  Presentation
```

| Layer | Owner code | Purpose |
|-------|------------|---------|
| **1. Ingestion** | `backend/app/ingestion/` | Parse PDFs, transcribe audio, chunk on legal boundaries. Runs async on ARQ. |
| **2. Processing** | `backend/app/retrieval/embeddings.py` + `vector_store.py` | BGE-M3 embeddings written to embedded Qdrant. Metadata written to Postgres. |
| **3. Retrieval** | `backend/app/retrieval/hybrid.py` + `reranker.py` + `backend/app/llm/` | Hybrid dense + BM25 fusion → cross-encoder rerank → LLM synthesis with citations. |
| **4. Audit (Report Shield)** | `backend/app/shield/` | Reverse-RAG. Splits an uploaded report into claims, verifies each against the case corpus. |
| **5. Presentation** | `frontend/` (SvelteKit) served by FastAPI | Upload zone, chat, evidence cards, Trust Score dashboard. |

## 2. Runtime topology

Three Docker services on the client machine:

```
┌─────────────┐   ┌───────────┐   ┌──────────────┐
│  postgres   │   │   redis   │   │   backend    │
│ metadata +  │   │  ARQ q +  │   │  FastAPI +   │
│ audit log   │   │ rate lim  │   │ embedded Qd  │
└─────────────┘   └───────────┘   └──────────────┘
                                        │
                                        │ static files
                                        ▼
                                  SvelteKit build/
```

- **Qdrant** runs *inside* the Python process (embedded mode). No separate container.
- **ARQ** runs inside FastAPI's async event loop. No separate worker container.
- **Frontend** is compiled once at build time into `frontend/build/` and mounted at `/` by FastAPI. No Node runtime shipped.

## 3. Request lifecycle — query path

```
User types question in SvelteKit
  → POST /v1/cases/{id}/query  (JWT in Authorization header)
    → Auth middleware validates JWT, checks case.org_id
      → RAG service:
          1. BGE-M3 embed(question)
          2. Qdrant top-30 filtered by case_id
          3. BM25 top-30 on same chunk corpus
          4. Fusion (RRF, dedup by chunk_id)
          5. Cross-encoder rerank → top 6
          6. LLM synthesize with citation-enforcing prompt
          7. Post-check: every citation must reference a chunk_id from step 6
        ← {answer, evidence[], confidence, latency_ms}
    ← 200 OK
  ← Streamed to UI with evidence cards
```

## 4. Request lifecycle — ingestion path

```
User drops PDF/audio in SvelteKit
  → POST /v1/cases/{id}/documents  (multipart)
    → Validate MIME + size + sanitize filename
    → Write to disk, compute sha256, dedup check
    → Enqueue ARQ task parse_document(doc_id)
    ← 200 OK  {document_id, task_id}  (< 200ms)

  [background]
  ARQ worker picks up task:
    → PyMuPDF extract text per page
    → pdfplumber extract tables separately
    → Clause-aware chunker (splits on '^\d+(\.\d+)*\s', section headings)
    → For each chunk: BGE-M3 embed → Qdrant insert with payload {case_id, doc_id, page, section}
    → Insert row into chunks table
    → UPDATE documents SET status = 'done'
```

## 5. Data flow — Report Shield (reverse RAG)

```
User uploads report (must already be ingested as a document)
  → POST /v1/cases/{id}/shield  {source_doc_id}
    → Load source document's chunks
    → Split into claims (numbered clauses / sentences)
    → For each claim:
        a. Run retrieval pipeline against the SAME case corpus
        b. LLM judges: supports / neutral / contradicts
        c. Label: Verified (≥0.7 conf) / Unverified / Contradicted
    → Missing-section check against doc-type template (contract / audit / RERA)
    → Aggregate: trust_score = verified_count / total_claims
    → Persist to shield_reports table
    ← {trust_score, flags[], missing_sections[]}
```

## 6. Security boundaries

- **JWT** on every non-auth endpoint. Missing → 401. Wrong org → 403.
- **case_id** is filtered at Qdrant query time (server-side, never trust client).
- **Prompt injection**: system prompt lockdown + output scanner. See TRD §12.4.
- **Audit log** is append-only. No admin endpoint deletes rows.

## 7. Tier switching

One env var `MODEL_TIER ∈ {1, 2, 3}` swaps embedding model, LLM, and vector store. Loaded once at startup via `backend/app/llm/model_selector.py`. See TRD §11 for the tier table.

## 8. What is NOT in v1

- Multi-user real-time collaboration on one case
- Web-hosted SaaS mode (v1 is client-hosted only)
- Automatic OCR of scanned PDFs (detected + warned only)
- Third-party telemetry (Sentry, DataDog) — everything stays local
