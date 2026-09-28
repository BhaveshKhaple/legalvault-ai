"""
Task 6.1 — FastAPI application entrypoint.

Run with: uvicorn backend.app.main:app --reload --port 8000
Browse:   http://localhost:8000/docs  (auto-generated Swagger UI)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import cases, documents, query, shield, auth

app = FastAPI(
    title="LegalVault AI",
    version="0.1.0",
    description="On-premise legal document retrieval and verification system.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # SvelteKit dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route prefixes match api_contract.md
app.include_router(auth.router,      prefix="/v1/auth",  tags=["auth"])
app.include_router(cases.router,     prefix="/v1/cases", tags=["cases"])
app.include_router(documents.router, prefix="/v1/cases", tags=["documents"])
app.include_router(query.router,     prefix="/v1/cases", tags=["query"])
app.include_router(shield.router,    prefix="/v1/cases", tags=["shield"])


@app.get("/healthz", tags=["ops"])
def health_check():
    return {"status": "ok"}
