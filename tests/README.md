# tests/

Layout mirrors `backend/app/`. If your code lives at `backend/app/foo/bar.py`, its tests live at `tests/foo/test_bar.py`.

## Planned layout

```
tests/
├── unit/                # pure-function tests (chunker, embedder, reranker wrappers)
├── integration/         # spin up postgres + redis in Docker, real DB calls
├── e2e/                 # scripted flow: upload → poll → query → shield → export
└── eval/                # RecallAt5 + Shield precision benchmarks
    ├── test_queries.json
    └── sample_reports/  # planted-flaw PDFs
```

## Run

```bash
cd backend && venv\Scripts\activate
pytest tests/unit -v
pytest tests/integration -v         # requires Docker up
pytest tests/e2e -v                 # requires full docker-compose up
```

## Coverage targets (from TRD §14)

- Unit tests: ≥60% coverage on `backend/app/`
- Integration: happy-path ingestion + query + Shield
- E2E: runs in CI on merge to `main`
- Eval: Recall@5 > 0.75, Shield precision > 80%
