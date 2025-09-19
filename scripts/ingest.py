"""CLI to load Torob parquet exports into Postgres."""

import argparse
import pathlib

import polars as pl
from polars.datatypes import List as PlList


def _normalise_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce problematic column types so they round-trip through ADBC.

    The Torob parquet exports contain a mix of null-only columns, struct-typed
    JSON blobs, and list columns whose inner types vary by table. Newer Polars
    releases tightened the type checks inside ``DataFrame.write_database`` which
    causes ingestion to fail unless we explicitly coerce those columns to types
    Postgres understands. This helper mirrors the logic previously inlined in
    :func:`load_table` but adds explicit handling for struct columns so the
    ingestion pipeline stays compatible with the latest Polars releases.

    Args:
        df: DataFrame constructed from a parquet file.

    Returns:
        The DataFrame with columns cast to database-friendly types.
    """

    transforms: list[pl.Expr] = []

    for name, dtype in df.schema.items():
        column = pl.col(name)

        if dtype == pl.Null:
            transforms.append(column.cast(pl.Utf8, strict=False).alias(name))
            continue

        if dtype == pl.Struct:
            transforms.append(
                pl.when(column.is_not_null())
                .then(column.struct.json_encode())
                .otherwise(None)
                .alias(name)
            )
            continue

        if isinstance(dtype, PlList):
            inner_type = dtype.inner
            if inner_type == pl.Null:
                transforms.append(
                    column.cast(pl.List(pl.Utf8), strict=False).alias(name)
                )
            elif inner_type == pl.Struct:
                transforms.append(
                    column.list.eval(
                        pl.when(pl.element().is_not_null())
                        .then(pl.element().struct.json_encode())
                        .otherwise(None)
                    ).alias(name)
                )

    if transforms:
        df = df.with_columns(transforms)

    return df
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
    df = _normalise_dataframe(pl.read_parquet(parquet_path))

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
