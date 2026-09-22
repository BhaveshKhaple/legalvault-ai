# backend/

FastAPI application. See `../docs/TRD.md` §3 for stack rationale and §5 for the API surface.

## Planned layout

```
backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── models.py               # SQLModel schemas
│   ├── routers/                # /v1/* route handlers
│   ├── ingestion/              # PDF, audio, chunker (Module 1)
│   ├── retrieval/              # embeddings, vector store, BM25, rerank (Modules 2-3)
│   ├── llm/                    # Ollama client, tier selector, prompts (Module 4)
│   ├── shield/                 # Reverse-RAG audit engine (Module 5)
│   ├── export/                 # Evidence ZIP + PDF report (Module 8)
│   ├── auth/                   # JWT, roles, dependencies
│   └── workers/                # ARQ background tasks
├── alembic/                    # DB migrations
├── requirements.txt
├── .env.example
├── venv/                       # local — gitignored
└── Dockerfile                  # from Task 9.1
```

## Setup

```bash
cd backend
py -3.11 -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

Then follow the task-specific setup notes in each PR.
