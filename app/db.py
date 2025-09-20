"""Database engine and session management utilities."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import settings
from . import models  # noqa: F401 - ensure models are imported for metadata

engine = create_async_engine(settings.async_database_url, future=True)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Provide an async SQLAlchemy session for FastAPI dependencies."""

    async with AsyncSessionLocal() as session:
        yield session
