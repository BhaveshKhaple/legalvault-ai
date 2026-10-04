"""
Task 6.1 — SQLModel data models (database tables).

Five core tables from TRD §4. All PKs are UUIDs. All FKs have ON DELETE
CASCADE unless noted. Alembic manages migrations — do NOT change column
types or names without a migration.

Run migrations: alembic upgrade head
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── enums ────────────────────────────────────────────────────────────────────


class DocType(str, Enum):
    contract = "contract"
    audit_report = "audit_report"
    agreement = "agreement"
    other = "other"


class FileType(str, Enum):
    pdf = "pdf"
    audio = "audio"


class IngestStatus(str, Enum):
    pending = "pending"
    indexing = "indexing"
    done = "done"
    error = "error"


class UserRole(str, Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class AuditAction(str, Enum):
    login = "login"
    upload = "upload"
    query = "query"
    shield_run = "shield_run"
    export = "export"
    delete = "delete"


# ─── users ────────────────────────────────────────────────────────────────────


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(max_length=200, unique=True, index=True)
    password_hash: str = Field(max_length=200)
    role: UserRole = Field(default=UserRole.analyst)
    org_id: Optional[uuid.UUID] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=_utcnow)


class Organization(SQLModel, table=True):
    __tablename__ = "organizations"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=200)
    created_at: datetime = Field(default_factory=_utcnow)


# ─── cases ────────────────────────────────────────────────────────────────────


class Case(SQLModel, table=True):
    __tablename__ = "cases"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=200)
    client_name: Optional[str] = Field(default=None, max_length=200)
    doc_type: DocType = Field(default=DocType.other)
    # Optional until Task 6.4 wires auth (SQLite doesn't enforce FKs by default)
    created_by: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id")
    org_id: Optional[uuid.UUID] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=_utcnow)


# ─── documents ────────────────────────────────────────────────────────────────


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    case_id: uuid.UUID = Field(foreign_key="cases.id")
    filename: str = Field(max_length=500)
    doc_type: FileType
    page_count: Optional[int] = Field(default=None)
    duration_sec: Optional[float] = Field(default=None)
    storage_path: str = Field(max_length=1000)
    sha256: str = Field(max_length=64, index=True)
    status: IngestStatus = Field(default=IngestStatus.pending)
    error_message: Optional[str] = Field(default=None)
    # embedding-dispatcher: which Qdrant collection holds this doc's vectors.
    # "legalvault_e5" (default) or "legalvault_bge". Needed at query time for
    # fan-out: RAG service reads this to build the vectors_by_collection map.
    qdrant_collection: str = Field(default="legalvault_e5", max_length=50)
    # Human-readable dispatcher decision ("e5_small" | "bge_m3")
    embedding_tier: str = Field(default="e5_small", max_length=20)
    created_at: datetime = Field(default_factory=_utcnow)


# ─── chunks ───────────────────────────────────────────────────────────────────


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id")
    content: str
    chunk_index: int
    page_number: Optional[int] = Field(default=None)
    ts_start: Optional[float] = Field(default=None)
    ts_end: Optional[float] = Field(default=None)
    section_title: Optional[str] = Field(default=None, max_length=500)
    qdrant_id: Optional[uuid.UUID] = Field(default=None)
    # Phase 2 — parent-child chunking.
    # chunk_role: "child" (embedded, retrieval) | "parent" (SQLite-only, LLM context)
    # Audio and legacy flat chunks default to "child" so queries still work.
    chunk_role: str = Field(default="child", max_length=10)
    # Pointer from a child to its parent chunk (same table). Null for parents
    # themselves, for audio chunks, and for pre-Phase-2 legacy rows.
    parent_chunk_id: Optional[uuid.UUID] = Field(default=None, foreign_key="chunks.id")
    # True when a chunk represents a table (atomic parent==child).
    is_table: bool = Field(default=False)


# ─── queries ──────────────────────────────────────────────────────────────────


class Query(SQLModel, table=True):
    __tablename__ = "queries"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    case_id: uuid.UUID = Field(foreign_key="cases.id")
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id")
    query_text: str
    answer_text: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)


# ─── shield reports ───────────────────────────────────────────────────────────


class ShieldReport(SQLModel, table=True):
    __tablename__ = "shield_reports"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    case_id: uuid.UUID = Field(foreign_key="cases.id")
    source_doc_id: uuid.UUID = Field(foreign_key="documents.id")
    trust_score: float
    verified_count: int = Field(default=0)
    unverified_count: int = Field(default=0)
    contradiction_count: int = Field(default=0)
    missing_sections: str = Field(default="[]")  # JSON array
    flags: str = Field(default="[]")             # JSON array
    created_at: datetime = Field(default_factory=_utcnow)


# ─── audit log (append-only) ──────────────────────────────────────────────────


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id")
    action: AuditAction
    resource_id: Optional[uuid.UUID] = Field(default=None)
    ip_addr: Optional[str] = Field(default=None, max_length=45)
    user_agent: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
