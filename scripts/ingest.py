"""CLI to load Torob parquet exports into Postgres."""

import argparse
import pathlib

import polars as pl
from polars.datatypes import List as PlList
import psycopg

TABLE_FILE_MAP = {
    "base_products": "base_products.parquet",
    "base_views": "base_views.parquet",
    "brands": "brands.parquet",
    "categories": "categories.parquet",
    "cities": "cities.parquet",
    "final_clicks": "final_clicks.parquet",
    "members": "members.parquet",
    "searches": "searches.parquet",
    "shops": "shops.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing parquet files from Torob dump.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        required=True,
        help="Postgres connection string (postgresql://user:pass@host/db)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="public",
        help="Target database schema for load operations.",
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="*",
        default=list(TABLE_FILE_MAP.keys()),
        help="Optional subset of tables to ingest.",
    )
    parser.add_argument(
        "--if-exists",
        choices=["replace", "append"],
        default="replace",
        help="Behaviour when target table already exists.",
    )
    return parser.parse_args()


def load_table(
    table: str,
    parquet_path: pathlib.Path,
    database_url: str,
    schema: str,
    if_exists: str,
) -> None:
    df = pl.read_parquet(parquet_path)

    transforms = []
    for name, dtype in df.schema.items():
        if dtype == pl.Null:
            transforms.append(pl.col(name).cast(pl.Utf8, strict=False))
        elif isinstance(dtype, PlList) and dtype.inner == pl.Null:
            transforms.append(pl.col(name).cast(pl.List(pl.Utf8), strict=False))

    if transforms:
        df = df.with_columns(transforms)

    table_name = f"{schema}.{table}" if schema else table
    df.write_database(
        table_name=table_name,
        connection=database_url,
        if_table_exists=if_exists,
        engine="adbc",
    )


def main() -> None:
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist")

    with psycopg.connect(args.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for table in args.tables:
        if table not in TABLE_FILE_MAP:
            raise SystemExit(f"Unknown table '{table}'")
        parquet_file = data_dir / TABLE_FILE_MAP[table]
        if not parquet_file.exists():
            raise SystemExit(f"Missing parquet file: {parquet_file}")
        load_table(table, parquet_file, args.database_url, args.schema, args.if_exists)
        print(f"Loaded {table}")


if __name__ == "__main__":
    main()
