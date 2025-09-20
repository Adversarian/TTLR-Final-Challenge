"""Utilities for loading parquet data into PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pyarrow.parquet as pq
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from .db import AsyncSessionLocal
from . import models

Row = Dict[str, Any]
TransformFn = Callable[[Row], Row | None]


def _to_python(value: Any) -> Any:
    """Convert Arrow scalars to native Python values."""

    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _maybe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _maybe_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def base36_decode(token: str | None) -> int | None:
    """Decode base36 identifiers emitted by the dataset."""

    if not token:
        return None
    return int(token, 36)


async def insert_chunk(session: AsyncSession, table, rows: List[Row]) -> int:
    """Insert a chunk of rows using PostgreSQL upsert semantics."""

    if not rows:
        return 0

    stmt = pg_insert(table).values(rows)
    pk_columns = [column.name for column in table.primary_key.columns]
    if pk_columns:
        stmt = stmt.on_conflict_do_nothing(index_elements=pk_columns)

    await session.execute(stmt)
    return len(rows)


async def load_parquet(
    session: AsyncSession,
    *,
    path: Path,
    table,
    chunk_size: int,
    transform: TransformFn,
) -> int:
    """Stream a parquet file into the given database table."""

    parquet_file = pq.ParquetFile(path)
    total_inserted = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk: List[Row] = []
        for raw_row in batch.to_pylist():
            row = {key: _to_python(value) for key, value in raw_row.items()}
            transformed = transform(row)
            if transformed:
                chunk.append(transformed)

        try:
            inserted = await insert_chunk(session, table, chunk)
        except ProgrammingError as exc:  # pragma: no cover - defensive guard
            sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", "")
            if sqlstate == "42P01":
                raise RuntimeError(
                    "Database schema missing required tables. Run migrations first."
                ) from exc
            raise

        if inserted:
            total_inserted += inserted
            await session.commit()

    return total_inserted


def _transform_city(row: Row) -> Row:
    return {"id": int(row["id"]), "name": row["name"]}


def _transform_brand(row: Row) -> Row:
    return {"id": int(row["id"]), "title": row["title"]}


def _transform_category(row: Row) -> Row:
    parent_id = _maybe_none(row.get("parent_id"))
    parent = int(parent_id) if parent_id is not None else -1
    return {"id": int(row["id"]), "title": row["title"], "parent_id": parent}


def _transform_shop(row: Row) -> Row:
    score_value = _maybe_none(row.get("score"))
    if score_value is None:
        raise ValueError(f"Missing score for shop id={row.get('id')}")

    has_warranty = row.get("has_warranty")
    if isinstance(has_warranty, str):
        has_warranty = has_warranty.strip().lower() in {"true", "t", "1"}
    elif has_warranty is None:
        has_warranty = False

    return {
        "id": int(row["id"]),
        "city_id": int(row["city_id"]),
        "score": float(score_value),
        "has_warranty": bool(has_warranty),
    }


def _transform_base_product(row: Row) -> Row:
    brand_id = _maybe_none(row.get("brand_id"))
    return {
        "random_key": row["random_key"],
        "persian_name": row["persian_name"],
        "english_name": _maybe_none(row.get("english_name")),
        "category_id": int(row["category_id"]),
        "brand_id": int(brand_id) if brand_id is not None else None,
        "extra_features": _maybe_json(row.get("extra_features"), {}),
        "image_url": _maybe_none(row.get("image_url")),
    }


def _transform_member(row: Row) -> Row:
    return {
        "random_key": row["random_key"],
        "base_random_key": row["base_random_key"],
        "shop_id": int(row["shop_id"]),
        "price": int(row["price"]),
    }


def _transform_search(row: Row) -> Row:
    return {
        "id": base36_decode(row.get("id")),
        "uid": row["uid"],
        "query": row["query"],
        "page": int(row["page"]),
        "timestamp": row.get("timestamp"),
        "session_id": row["session_id"],
        "result_base_product_rks": _maybe_json(row.get("result_base_product_rks"), []),
        "category_id": int(row.get("category_id", 0)),
        "category_brand_boosts": _maybe_json(row.get("category_brand_boosts"), []),
    }


def _transform_base_view(row: Row) -> Row:
    return {
        "id": base36_decode(row.get("id")),
        "search_id": base36_decode(row.get("search_id")),
        "base_product_rk": row["base_product_rk"],
        "timestamp": row.get("timestamp"),
    }


def _transform_final_click(row: Row) -> Row:
    return {
        "id": base36_decode(row.get("id")),
        "base_view_id": base36_decode(row.get("base_view_id")),
        "shop_id": int(row["shop_id"]),
        "timestamp": row.get("timestamp"),
    }


TABLE_LOADERS: Dict[str, tuple[str, Any, TransformFn]] = {
    "cities": ("cities.parquet", models.City.__table__, _transform_city),
    "brands": ("brands.parquet", models.Brand.__table__, _transform_brand),
    "categories": ("categories.parquet", models.Category.__table__, _transform_category),
    "shops": ("shops.parquet", models.Shop.__table__, _transform_shop),
    "base_products": (
        "base_products.parquet",
        models.BaseProduct.__table__,
        _transform_base_product,
    ),
    "members": ("members.parquet", models.Member.__table__, _transform_member),
    "searches": ("searches.parquet", models.Search.__table__, _transform_search),
    "base_views": ("base_views.parquet", models.BaseView.__table__, _transform_base_view),
    "final_clicks": (
        "final_clicks.parquet",
        models.FinalClick.__table__,
        _transform_final_click,
    ),
}

DEFAULT_LOAD_ORDER: List[str] = [
    "cities",
    "brands",
    "categories",
    "shops",
    "base_products",
    "members",
    "searches",
    "base_views",
    "final_clicks",
]


async def load_table(
    session: AsyncSession,
    *,
    data_dir: Path,
    table_name: str,
    chunk_size: int,
) -> int:
    """Load a single table from its parquet file."""

    if table_name not in TABLE_LOADERS:
        raise ValueError(f"Unknown table requested: {table_name}")

    filename, table, transform = TABLE_LOADERS[table_name]
    path = (data_dir / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    return await load_parquet(
        session,
        path=path,
        table=table,
        chunk_size=chunk_size,
        transform=transform,
    )


async def load_all_tables(
    data_dir: Path,
    *,
    chunk_size: int = 1_000,
    tables: Iterable[str] | None = None,
) -> Dict[str, int]:
    """Load all requested tables and return a mapping of inserted row counts."""

    requested = list(tables) if tables else DEFAULT_LOAD_ORDER
    available = set(DEFAULT_LOAD_ORDER)
    unknown = [table for table in requested if table not in available]
    if unknown:
        raise ValueError(f"Unknown tables requested: {', '.join(unknown)}")

    ordered_tables = [table for table in DEFAULT_LOAD_ORDER if table in requested]

    results: Dict[str, int] = {}
    async with AsyncSessionLocal() as session:
        for table_name in ordered_tables:
            inserted = await load_table(
                session,
                data_dir=data_dir,
                table_name=table_name,
                chunk_size=chunk_size,
            )
            results[table_name] = inserted

    return results


def load_all_tables_sync(
    data_dir: Path,
    *,
    chunk_size: int = 1_000,
    tables: Iterable[str] | None = None,
) -> Dict[str, int]:
    """Synchronous wrapper around :func:`load_all_tables`."""

    return asyncio.run(load_all_tables(data_dir, chunk_size=chunk_size, tables=tables))
