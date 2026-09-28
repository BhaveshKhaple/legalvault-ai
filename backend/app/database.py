"""
Task 6.1 — Database session factory.

Uses SQLModel + SQLAlchemy async engine. FastAPI route handlers get a
database session via the get_session() dependency.
"""

import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://legalvault:legalvault@localhost:5432/legalvault",
)

# asyncpg driver required for async FastAPI; psycopg2 used by Alembic (sync)
engine = create_async_engine(_DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session():
    """FastAPI dependency: yields a database session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def create_db_and_tables():
    """Create all tables (dev only — prod uses Alembic migrations)."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
