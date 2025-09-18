from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

_pool: ConnectionPool | None = None


def init_pool(conninfo: str, *, min_size: int = 1, max_size: int = 5) -> None:
    global _pool
    if not conninfo:
        raise ValueError("DATABASE_URL is required to initialise the pool")
    if _pool is None:
        _pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size)
    else:
        if _pool.closed:
            _pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size)


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Connection pool has not been initialised")
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_cursor():
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            yield cursor
