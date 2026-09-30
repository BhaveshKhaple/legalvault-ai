"""
Task 6.1 — FastAPI application entrypoint.
Task 6.2 — Added lifespan (DB init on startup) + static file serving for UI.

Run with:  uvicorn backend.app.main:app --reload --port 8000
Browse:    http://localhost:8000/         (UI)
Swagger:   http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.database import create_db_and_tables
from backend.app.routers import auth, cases, documents, query, shield


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(
    title="LegalVault AI",
    version="0.1.0",
    description="On-premise legal document retrieval and verification system.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev — tighten via env var in prod
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
    return {"status": "ok", "version": app.version}


# Serve HTML UI — mounted last so /v1/* API routes take precedence
_FRONTEND_DIR = Path(__file__).parents[3] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
