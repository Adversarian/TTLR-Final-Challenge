"""Database access helpers for Postgres."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping, Optional

import asyncpg


class DatabasePool:
    """Singleton-like asyncpg pool manager."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()

    async def connect(self, dsn: str) -> None:
        """Initialise the pool if it has not been created yet."""

        if self._pool is not None:
            return
        async with self._lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        """Close the connection pool."""

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Yield a pooled connection."""

        if self._pool is None:
            raise RuntimeError("Database pool has not been initialised")
        async with self._pool.acquire() as conn:  # type: ignore[async-with-compat]
            yield conn

    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[asyncpg.Record]:
        """Fetch multiple rows from the database."""

        if self._pool is None:
            raise RuntimeError("Database pool has not been initialised")
        async with self.connection() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> Optional[asyncpg.Record]:
        """Fetch a single row from the database."""

        if self._pool is None:
            raise RuntimeError("Database pool has not been initialised")
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        """Execute a statement returning its status."""

        if self._pool is None:
            raise RuntimeError("Database pool has not been initialised")
        async with self.connection() as conn:
            return await conn.execute(query, *args, timeout=timeout)


db_pool = DatabasePool()
"""Global database pool instance."""
