"""Dataset ingestion script to populate Postgres from parquet payloads."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import asyncpg
import gdown
import pandas as pd

from .services.feature_utils import normalize_unit
from .services.preprocess import normalize_query

DATA_URL = "https://drive.google.com/uc?id=1W4mSI33IbeKkWztK3XmE05F7m4tNYDYu"


async def create_extensions(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent")


async def create_tables(conn: asyncpg.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS brands (
            id BIGINT PRIMARY KEY,
            title TEXT
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id BIGINT PRIMARY KEY,
            title TEXT,
            parent_id BIGINT
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shops (
            id BIGINT PRIMARY KEY,
            city_id BIGINT,
            score NUMERIC,
            has_warranty BOOLEAN
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS base_products (
            random_key TEXT PRIMARY KEY,
            persian_name TEXT,
            english_name TEXT,
            category_id BIGINT,
            brand_id BIGINT,
            brand_title TEXT,
            category_path TEXT,
            extra_features JSONB,
            extra_features_flat TEXT,
            search_text TEXT,
            image_url TEXT
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS members (
            random_key TEXT PRIMARY KEY,
            base_random_key TEXT,
            shop_id BIGINT,
            price NUMERIC
        )
        """
    )


async def create_indexes(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE INDEX IF NOT EXISTS base_products_fts ON base_products USING GIN (to_tsvector('simple', search_text))")
    await conn.execute("CREATE INDEX IF NOT EXISTS base_products_trgm ON base_products USING GIN (search_text gin_trgm_ops)")
    await conn.execute("CREATE INDEX IF NOT EXISTS members_base ON members(base_random_key)")
    await conn.execute("CREATE INDEX IF NOT EXISTS members_base_price ON members(base_random_key, price)")
    await conn.execute("CREATE INDEX IF NOT EXISTS shops_city ON shops(city_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS shops_warranty ON shops(has_warranty)")


def download_dataset(destination: Path, url: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    gdown.download(url, str(destination), quiet=False, fuzzy=True)
    return destination


def extract_archive(archive_path: Path) -> Path:
    target_dir = Path(tempfile.mkdtemp(prefix="ttlrdataset_"))
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target_dir)
    return target_dir


def flatten_features(features: Dict) -> str:
    tokens: List[str] = []
    for key, value in features.items():
        if isinstance(value, dict):
            raw = value.get("value")
            text = value.get("text")
            unit = value.get("unit")
            if raw is not None:
                tokens.append(f"{key}:{raw}")
                if isinstance(raw, (int, float)):
                    tokens.append(f"{key}__num={raw}")
            if text:
                tokens.append(f"{key}:{text}")
            if unit:
                tokens.append(f"{key}:{normalize_unit(unit)}")
        else:
            tokens.append(f"{key}:{value}")
            if isinstance(value, (int, float)):
                tokens.append(f"{key}__num={value}")
    return " ".join(str(t) for t in tokens)


def build_category_paths(categories: pd.DataFrame) -> Dict[int, str]:
    lookup = {int(row.id): (row.title, int(row.parent_id) if row.parent_id not in (-1, None) else None) for row in categories.itertuples()}
    cache: Dict[int, str] = {}

    def compute(cat_id: int) -> str:
        if cat_id in cache:
            return cache[cat_id]
        title, parent = lookup.get(cat_id, ("", None))
        if parent is None:
            path = title or ""
        else:
            parent_path = compute(parent)
            path = " / ".join(filter(None, [parent_path, title]))
        cache[cat_id] = path
        return path

    for cat_id in lookup:
        compute(cat_id)
    return cache


def prepare_base_products(df: pd.DataFrame, brands: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    brand_lookup = {int(row.id): row.title for row in brands.itertuples()}
    category_paths = build_category_paths(categories)
    df = df.copy()
    df["brand_title"] = df["brand_id"].map(brand_lookup).fillna("")
    df["category_path"] = df["category_id"].map(category_paths).fillna("")
    df["extra_features"] = df["extra_features"].apply(lambda v: v if isinstance(v, dict) else {})
    df["extra_features_flat"] = df["extra_features"].apply(flatten_features)

    def make_search_text(row: pd.Series) -> str:
        parts = [
            row.get("persian_name", ""),
            row.get("english_name", ""),
            row.get("brand_title", ""),
            row.get("category_path", ""),
            row.get("extra_features_flat", ""),
        ]
        joined = " ".join(str(part) for part in parts if part)
        return normalize_query(joined)

    df["search_text"] = df.apply(make_search_text, axis=1)
    return df


async def copy_dataframe(conn: asyncpg.Connection, table: str, columns: Iterable[str], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    records: List[Tuple] = []
    for row in frame.itertuples(index=False, name=None):
        records.append(tuple(row))
    await conn.copy_records_to_table(table, records=records, columns=list(columns))


async def ingest(database_url: str, dataset_dir: Path) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        await create_extensions(conn)
        await create_tables(conn)

        brands = pd.read_parquet(dataset_dir / "brands.parquet")
        categories = pd.read_parquet(dataset_dir / "categories.parquet")
        shops = pd.read_parquet(dataset_dir / "shops.parquet")
        members = pd.read_parquet(dataset_dir / "members.parquet")
        base_products = pd.read_parquet(dataset_dir / "base_products.parquet")

        base_products_prepared = prepare_base_products(base_products, brands, categories)

        await conn.execute("TRUNCATE brands RESTART IDENTITY CASCADE")
        await conn.execute("TRUNCATE categories RESTART IDENTITY CASCADE")
        await conn.execute("TRUNCATE shops RESTART IDENTITY CASCADE")
        await conn.execute("TRUNCATE members RESTART IDENTITY CASCADE")
        await conn.execute("TRUNCATE base_products RESTART IDENTITY CASCADE")

        await copy_dataframe(
            conn,
            "brands",
            ["id", "title"],
            brands[["id", "title"]],
        )
        await copy_dataframe(
            conn,
            "categories",
            ["id", "title", "parent_id"],
            categories[["id", "title", "parent_id"]],
        )
        await copy_dataframe(
            conn,
            "shops",
            ["id", "city_id", "score", "has_warranty"],
            shops[["id", "city_id", "score", "has_warranty"]],
        )
        await copy_dataframe(
            conn,
            "members",
            ["random_key", "base_random_key", "shop_id", "price"],
            members[["random_key", "base_random_key", "shop_id", "price"]],
        )

        base_subset = base_products_prepared[
            [
                "random_key",
                "persian_name",
                "english_name",
                "category_id",
                "brand_id",
                "brand_title",
                "category_path",
                "extra_features",
                "extra_features_flat",
                "search_text",
                "image_url",
            ]
        ]
        base_subset = base_subset.apply(
            lambda row: (
                row["random_key"],
                row["persian_name"],
                row["english_name"],
                row["category_id"],
                row["brand_id"],
                row["brand_title"],
                row["category_path"],
                json.dumps(row["extra_features"] or {}),
                row["extra_features_flat"],
                row["search_text"],
                row.get("image_url"),
            ),
            axis=1,
        )
        base_records = pd.DataFrame(base_subset.tolist(), columns=[
            "random_key",
            "persian_name",
            "english_name",
            "category_id",
            "brand_id",
            "brand_title",
            "category_path",
            "extra_features",
            "extra_features_flat",
            "search_text",
            "image_url",
        ])
        await copy_dataframe(
            conn,
            "base_products",
            base_records.columns,
            base_records,
        )

        await create_indexes(conn)
    finally:
        await conn.close()


async def async_main(database_url: str, url: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = download_dataset(tmp_path / "dataset.tar.gz", url)
        dataset_dir = extract_archive(archive_path)
        await ingest(database_url, dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TTLR dataset into Postgres")
    parser.add_argument("--database", default=os.getenv("DATABASE_URL"), help="Postgres connection URI")
    parser.add_argument("--url", default=os.getenv("DATA_URL", DATA_URL), help="Dataset download URL")
    args = parser.parse_args()
    if not args.database:
        raise SystemExit("DATABASE_URL must be provided")
    asyncio.run(async_main(args.database, args.url))


if __name__ == "__main__":
    main()
