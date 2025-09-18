"""Populate pgvector embeddings for base products."""

import argparse
from typing import Iterable

import psycopg
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="Postgres connection string")
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="OpenAI embedding model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of products to embed per batch",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Recompute embeddings for all products (not only missing ones)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of products to embed",
    )
    return parser.parse_args()


def ensure_table(cur: psycopg.Cursor, dimension: int) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS product_embeddings (
            random_key TEXT PRIMARY KEY REFERENCES base_products(random_key) ON DELETE CASCADE,
            embedding vector(%s)
        )
        """,
        (dimension,),
    )


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def main() -> None:
    args = parse_args()
    client = OpenAI()

    with psycopg.connect(args.database_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            sample_embedding = client.embeddings.create(
                model=args.model, input="torob sample"
            ).data[0].embedding
            dimension = len(sample_embedding)
            ensure_table(cur, dimension)
            conn.commit()

        if args.refresh_all:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE product_embeddings")
            conn.commit()

        remaining = args.limit
        while remaining is None or remaining > 0:
            batch_size = args.batch_size if remaining is None else min(args.batch_size, remaining)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT bp.random_key,
                           coalesce(bp.persian_name,'') || ' ' || coalesce(bp.english_name,'') || ' ' || coalesce(bp.extra_features,'') AS text
                    FROM base_products bp
                    LEFT JOIN product_embeddings pe ON pe.random_key = bp.random_key
                    WHERE pe.random_key IS NULL
                    ORDER BY bp.random_key
                    LIMIT %s
                    """,
                    (batch_size,),
                )
                rows = cur.fetchall()

            if not rows:
                break

            texts = [row[1] or row[0] for row in rows]
            embeddings = client.embeddings.create(model=args.model, input=texts)

            with conn.cursor() as cur:
                for row, embed in zip(rows, embeddings.data):
                    literal = vector_literal(embed.embedding)
                    cur.execute(
                        """
                        INSERT INTO product_embeddings (random_key, embedding)
                        VALUES (%s, %s::vector)
                        ON CONFLICT (random_key) DO UPDATE SET embedding = EXCLUDED.embedding
                        """,
                        (row[0], literal),
                    )
            conn.commit()

            if remaining is not None:
                remaining -= len(rows)


if __name__ == "__main__":
    main()
