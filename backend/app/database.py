"""
Task 6.1 — Database session factory.
Task 6.2 — Switched default driver to SQLite+aiosqlite for local dev.

Set DATABASE_URL in .env to switch to Postgres for production:
  DATABASE_URL=postgresql+asyncpg://legalvault:legalvault@localhost:5432/legalvault

SQLite works for local testing without Docker. Alembic migrations still
target Postgres in CI/prod; SQLite dev uses create_db_and_tables() at startup.
"""

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/legalvault.db",
)

# SQLite needs check_same_thread=False for async; Postgres does not
_connect_args = {"check_same_thread": False} if _DATABASE_URL.startswith("sqlite") else {}

engine = create_async_engine(
    _DATABASE_URL,
    echo=False,
    future=True,
    connect_args=_connect_args,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session():
    """FastAPI dependency: yields a database session per request."""
    async with AsyncSessionLocal() as session:
        yield session


def _sync_create_and_migrate(connection):
    SQLModel.metadata.create_all(connection)
    if connection.dialect.name == "sqlite":
        from sqlalchemy import inspect, text

        inspector = inspect(connection)
        for table_name, table in SQLModel.metadata.tables.items():
            if inspector.has_table(table_name):
                existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
                for col in table.columns:
                    if col.name not in existing_cols:
                        col_type = col.type.compile(connection.dialect)
                        default_clause = ""
                        if col.default is not None and col.default.is_scalar:
                            val = col.default.arg
                            if isinstance(val, str):
                                default_clause = f" DEFAULT '{val}'"
                            elif isinstance(val, bool):
                                default_clause = f" DEFAULT {1 if val else 0}"
                            elif isinstance(val, (int, float)):
                                default_clause = f" DEFAULT {val}"
                        elif col.server_default is not None:
                            val = getattr(col.server_default, "arg", None)
                            if val is not None:
                                default_clause = f" DEFAULT {val}"
                        connection.execute(
                            text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}{default_clause}")
                        )


async def create_db_and_tables():
    """Create all tables on startup (dev mode — prod uses Alembic)."""
    Path("./data/documents").mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(_sync_create_and_migrate)

