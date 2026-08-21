# LegalVault AI

**Google Search for your legal documents, with cited proof and a report audit shield.**

An on-premise, evidence-first document assistant for Indian legal and BFSI teams. Ask questions in plain English across contracts and meeting recordings, get answers with exact source citations, and audit finished reports against source documents before you file them.

Runs offline. No cloud APIs. Data never leaves the client machine.

Built by **Team Pied Piper** as a final-year major project for DIPEX 2025.

---

## Status

This is an active WIP rewrite. The prior prototype (Streamlit + Gemini + FAISS) lives at [Multimodal-SIH](https://github.com/BhaveshKhaple/Multimodal-SIH). This repo is the ground-up v2 build.

- Bootstrap phase: in progress
- Target: working demo by DIPEX 2025
- See `PRD.md` for full product scope and `TRD.md` for architecture

## What it does

Two workflows, one system:

**Search mode** — Upload PDFs and audio recordings. Ask questions. Get answers with citations linking to exact PDF pages or audio timestamps.

**Report Shield mode** — Upload a finished audit or compliance report. The system extracts every claim, verifies each one against the case's source documents, and returns a Trust Score plus a gap list of unverified claims, contradictions, and missing standard clauses.

## Tech stack

| Layer | Choice |
|-------|--------|
| Backend | FastAPI + SQLModel + Alembic |
| RAG framework | LlamaIndex |
| Vector store | Qdrant (embedded) with pgvector fallback |
| Sparse index | BM25 (rank_bm25) |
| Embeddings | BGE-M3 (tier 1), E5-small (tier 2), EmbeddingGemma-300M (tier 3) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Qwen3 MoE / Phi-4 / Llama 3.2, all via Ollama |
| PDF | PyMuPDF + pdfplumber |
| Audio | openai-whisper (local) |
| Task queue | ARQ (Redis-backed) |
| Frontend | SvelteKit (compiled to static, served by FastAPI) |
| Storage | PostgreSQL + Redis |
| Deployment | Docker Compose, 3 services |

Every choice has a rationale documented in `TRD.md` §3.

## Repo structure

```
legalvault-ai/
├── backend/
│   ├── app.py                    # FastAPI entrypoint
│   ├── api/                      # route handlers (/v1/*)
│   ├── services/
│   │   ├── ingestion/            # PDF + Whisper + chunker
│   │   ├── retrieval/            # dense + BM25 + reranker
│   │   ├── llm/                  # Ollama wrappers, prompts
│   │   ├── report_shield/        # reverse-RAG audit engine
│   │   └── export/               # evidence ZIP, PDF report
│   ├── db/                       # SQLModel schemas, Alembic
│   └── tests/                    # unit + integration + e2e
├── frontend/                     # SvelteKit
├── configs/
│   ├── tiers.yaml                # tiered model config
│   └── templates/                # Report Shield doc-type templates
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs/
│   ├── updates/                  # per-task update.md files
│   └── architecture.md           # visual diagrams
├── PRD.md                        # Product Requirements
├── TRD.md                        # Technical Requirements
├── RULES.md                      # Team rules + AI coding guardrails
├── CONTRIBUTING.md               # PR flow (see below)
└── README.md
```

## Quick start (dev, once bootstrap is done)

```bash
# 1. Clone your fork
git clone git@github.com:<your-username>/legalvault-ai.git
cd legalvault-ai

# 2. Copy env template and fill values
cp .env.example .env

# 3. Boot the stack
docker compose up -d postgres redis
cd backend && pip install -r requirements.txt

# 4. Run migrations
alembic upgrade head

# 5. Start dev server
uvicorn app:app --reload --port 8000

# 6. Open UI
open http://localhost:8000
```

Full setup lives in `docs/setup.md` (after Task 0.1 ships).

## Development workflow

Sequential, fork-based. One task in flight at a time across the whole team. Rules and rationale are in `RULES.md`.

The short version:

```
1. Pick a Ready task from Major project Tracker.xlsx.
2. Set Status = In Progress. Set Owner = you.
3. Fork this repo. Clone your fork.
4. Branch: feature/<module>-<task-id>  e.g. feature/ingestion-1.2
5. Write code. Write update.md. Push to your fork.
6. Open PR against v1 branch.
7. Status = In Review. Bhavesh reviews.
8. Pass → merged to v1 → then main. Fail → fix and re-request review.
9. Done. Pick next Ready task.
```

Only ONE task may be "In Progress" at any time. Check the tracker Dashboard.

Read `RULES.md` before writing any code, especially if you plan to use AI assistance (Codex, Antigravity, Claude, Copilot). AI is welcome and encouraged. Unreviewed AI output is not.

## Team

| Role | Person |
|------|--------|
| Team Lead / Reviewer | Bhavesh Khaple ([@BhaveshKhaple](https://github.com/BhaveshKhaple)) |
| Developer | Shubham |
| Developer | Vijay |
| Developer | Tejas Bagal |

## Documentation

- `PRD.md` — Product Requirements Document (what we are building, for whom, and why)
- `TRD.md` — Technical Requirements Document (architecture, schema, APIs, security)
- `RULES.md` — Team development rules and AI coding guardrails
- `Major project Tracker.xlsx` — Sprint tracker (source of truth for task state)
- Pitch deck — DIPEX 2025 narrative (in shared Drive)

## Contributing

Only current team members contribute in v1. External contributions are welcome after DIPEX. See `CONTRIBUTING.md` (ships with Task 0.3).

## License

TBD before public release. Default: All Rights Reserved until team decides.

## Contact

Bhavesh Khaple — [bhaveshkhaple2@gmail.com](mailto:bhaveshkhaple2@gmail.com) — [LinkedIn](https://linkedin.com/in/bhavesh-khaple)
