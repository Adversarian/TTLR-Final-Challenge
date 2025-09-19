"""Populate pgvector embeddings for base products via LlamaIndex."""

import argparse
from typing import List
from urllib.parse import urlparse, parse_qs
import json
import psycopg
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", required=True, help="Postgres connection string"
    )
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


def _embed_dim_for(model: str) -> int:
    return 3072 if "text-embedding-3-large" in model else 1536


def _vector_store(database_url: str, embed_dim: int) -> PGVectorStore:
    parsed = urlparse(database_url)
    query = parse_qs(parsed.query)
    return PGVectorStore.from_params(
        database=parsed.path.lstrip("/"),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or query.get("user", [None])[0],
        password=parsed.password or query.get("password", [None])[0],
        table_name="product_embeddings",
        embed_dim=embed_dim,
        hybrid_search=True,
        text_search_config="simple",
    )


def _truncate_embeddings(database_url: str) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE product_embeddings")


def main() -> None:
    args = parse_args()
    embed_model = OpenAIEmbedding(model=args.model)
    store = _vector_store(args.database_url, _embed_dim_for(args.model))

    if args.refresh_all:
        _truncate_embeddings(args.database_url)

    remaining = args.limit
    while remaining is None or remaining > 0:
        batch_size = (
            args.batch_size if remaining is None else min(args.batch_size, remaining)
        )
        with psycopg.connect(args.database_url, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT bp.random_key,
                        bp.persian_name,
                        bp.english_name,
                        bp.extra_features,
                        bp.category_id,
                        bp.brand_id
                    FROM base_products bp
                    LEFT JOIN product_embeddings pe ON pe.doc_id = bp.random_key
                    WHERE pe.doc_id IS NULL
                    ORDER BY bp.random_key
                    LIMIT %s
                    """,
                    (batch_size,),
                )
                rows = cur.fetchall()

        if not rows:
            break

        documents: List[Document] = []
        for (
            random_key,
            persian_name,
            english_name,
            extra_features,
            category_id,
            brand_id,
        ) in rows:
            ef_str = None
            if extra_features is not None:
                ef_str = (
                    extra_features
                    if isinstance(extra_features, str)
                    else json.dumps(extra_features, ensure_ascii=False)
                )

            text_parts = [part for part in [persian_name, english_name, ef_str] if part]
            text = " \n".join(text_parts) if text_parts else random_key
            documents.append(
                Document(
                    id_=random_key,
                    text=text,
                    metadata={
                        "random_key": random_key,
                        "persian_name": persian_name,
                        "english_name": english_name,
                        "category_id": category_id,
                        "brand_id": brand_id,
                        "match_type": "semantic",
                    },
                )
            )

        store.add_documents(documents, embed_model=embed_model, show_progress=True)

        if remaining is not None:
            remaining -= len(documents)


if __name__ == "__main__":
    main()
