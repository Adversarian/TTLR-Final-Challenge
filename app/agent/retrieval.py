"""Database and vector-store helpers for assembling product context."""

from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from psycopg.rows import dict_row

from app.config import settings
from app.db import get_pool

from .models import ProductCandidate, ProductContext, ProductLookupArgs, SellerOffer, SellerStats


def _to_float(value: Optional[Decimal | float | int | str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, (int, Decimal)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_features(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}

    features: Dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value is None:
                continue
            features[str(key)] = str(value)
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = item.get("title") or item.get("name") or item.get("key")
            value = item.get("value") or item.get("val") or item.get("text")
            if not key or value in (None, ""):
                continue
            features[str(key)] = str(value)
    return features


@lru_cache(maxsize=1)
def _vector_index() -> VectorStoreIndex:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be configured for vector search")

    parsed = urlparse(settings.database_url)
    query = parse_qs(parsed.query)
    store = PGVectorStore.from_params(
        database=parsed.path.lstrip("/"),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or query.get("user", [None])[0],
        password=parsed.password or query.get("password", [None])[0],
        table_name="product_embeddings",
        hybrid_search=True,
        text_search_config="simple",
    )
    return VectorStoreIndex.from_vector_store(store)


def _hybrid_retrieve(product_name: str, top_k: int) -> List[ProductCandidate]:
    try:
        retriever = _vector_index().as_retriever(
            similarity_top_k=max(5, top_k),
            vector_store_kwargs={"hybrid_search": True},
        )
        nodes = retriever.retrieve(product_name)
    except Exception:
        return []

    matches: List[ProductCandidate] = []
    for node in nodes:
        metadata = node.metadata or {}
        random_key = metadata.get("random_key")
        if not random_key:
            continue
        score = getattr(node, "score", None)
        matches.append(
            ProductCandidate(
                random_key=random_key,
                persian_name=metadata.get("persian_name"),
                english_name=metadata.get("english_name"),
                matched_via=metadata.get("match_type", "hybrid"),
                score=float(score) if score is not None else None,
            )
        )
    return matches


def _sql_fuzzy_match(product_name: str, limit: int) -> List[ProductCandidate]:
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT bp.random_key,
                       bp.persian_name,
                       bp.english_name,
                       GREATEST(
                           similarity(COALESCE(bp.persian_name, ''), %s),
                           similarity(COALESCE(bp.english_name, ''), %s)
                       ) AS score
                FROM base_products bp
                ORDER BY score DESC NULLS LAST
                LIMIT %s
                """,
                (product_name, product_name, limit),
            )
            rows = cur.fetchall()

    matches: List[ProductCandidate] = []
    for row in rows:
        score = row["score"]
        if score is None or score <= 0:
            continue
        matches.append(
            ProductCandidate(
                random_key=row["random_key"],
                persian_name=row["persian_name"],
                english_name=row["english_name"],
                matched_via="lexical",
                score=float(score),
            )
        )
    return matches


def _fetch_product_by_random_key(random_key: str) -> Optional[ProductCandidate]:
    if not random_key:
        return None
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT bp.random_key, bp.persian_name, bp.english_name
                FROM base_products bp
                WHERE bp.random_key = %s
                """,
                (random_key,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return ProductCandidate(
        random_key=row["random_key"],
        persian_name=row["persian_name"],
        english_name=row["english_name"],
        matched_via="base_random_key",
        score=1.0,
    )


def _fetch_product_by_member_key(member_key: str) -> Optional[ProductCandidate]:
    if not member_key:
        return None
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT bp.random_key, bp.persian_name, bp.english_name
                FROM members m
                JOIN base_products bp ON bp.random_key = m.base_random_key
                WHERE m.random_key = %s
                """,
                (member_key,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return ProductCandidate(
        random_key=row["random_key"],
        persian_name=row["persian_name"],
        english_name=row["english_name"],
        matched_via="member_random_key",
        score=1.0,
    )


def _hydrate_product_contexts(candidates: Iterable[ProductCandidate]) -> List[ProductContext]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []

    ordered_keys: List[str] = []
    seen: set[str] = set()
    for candidate in candidate_list:
        if candidate.random_key not in seen:
            ordered_keys.append(candidate.random_key)
            seen.add(candidate.random_key)

    base_map: Dict[str, Dict[str, object]] = {}
    stats_map: Dict[str, SellerStats] = {}
    offers_map: Dict[str, List[SellerOffer]] = {}

    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT bp.random_key,
                       bp.persian_name,
                       bp.english_name,
                       bp.category_id,
                       c.title AS category_title,
                       bp.brand_id,
                       b.title AS brand_title,
                       bp.extra_features
                FROM base_products bp
                LEFT JOIN categories c ON c.id = bp.category_id
                LEFT JOIN brands b ON b.id = bp.brand_id
                WHERE bp.random_key = ANY(%s)
                """,
                (ordered_keys,),
            )
            for row in cur.fetchall():
                features = _parse_features(row["extra_features"])
                feature_list = [f"{k}: {v}" for k, v in features.items()]
                base_map[row["random_key"]] = {
                    "persian_name": row["persian_name"],
                    "english_name": row["english_name"],
                    "category_id": row["category_id"],
                    "category_title": row["category_title"],
                    "brand_id": row["brand_id"],
                    "brand_title": row["brand_title"],
                    "features": features,
                    "feature_list": feature_list,
                }

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT m.base_random_key,
                       COUNT(*) AS offer_count,
                       MIN(m.price) AS min_price,
                       MAX(m.price) AS max_price,
                       AVG(m.price) AS avg_price
                FROM members m
                WHERE m.base_random_key = ANY(%s)
                GROUP BY m.base_random_key
                """,
                (ordered_keys,),
            )
            for row in cur.fetchall():
                stats_map[row["base_random_key"]] = SellerStats(
                    offer_count=row["offer_count"] or 0,
                    min_price=_to_float(row["min_price"]),
                    max_price=_to_float(row["max_price"]),
                    avg_price=_to_float(row["avg_price"]),
                )

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT m.base_random_key,
                           m.random_key,
                           m.price,
                           m.shop_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY m.base_random_key
                               ORDER BY m.price ASC NULLS LAST, m.random_key
                           ) AS row_number
                    FROM members m
                    WHERE m.base_random_key = ANY(%s)
                )
                SELECT r.base_random_key,
                       r.random_key,
                       r.price,
                       r.shop_id,
                       s.score AS shop_score,
                       s.has_warranty,
                       s.city_id,
                       c.name AS city_name
                FROM ranked r
                LEFT JOIN shops s ON s.id = r.shop_id
                LEFT JOIN cities c ON c.id = s.city_id
                WHERE r.row_number <= 3
                ORDER BY r.base_random_key, r.row_number
                """,
                (ordered_keys,),
            )
            for row in cur.fetchall():
                offers_map.setdefault(row["base_random_key"], []).append(
                    SellerOffer(
                        member_random_key=row["random_key"],
                        price=_to_float(row["price"]),
                        shop_id=row["shop_id"],
                        shop_score=_to_float(row["shop_score"]),
                        has_warranty=row["has_warranty"],
                        city_id=row["city_id"],
                        city_name=row["city_name"],
                    )
                )

    contexts: List[ProductContext] = []
    emitted: set[str] = set()
    for candidate in candidate_list:
        if candidate.random_key in emitted:
            continue
        base_row = base_map.get(candidate.random_key, {})
        contexts.append(
            ProductContext(
                random_key=candidate.random_key,
                persian_name=base_row.get("persian_name") or candidate.persian_name,
                english_name=base_row.get("english_name") or candidate.english_name,
                category_id=base_row.get("category_id"),
                category_title=base_row.get("category_title"),
                brand_id=base_row.get("brand_id"),
                brand_title=base_row.get("brand_title"),
                matched_via=candidate.matched_via,
                match_score=candidate.score,
                features=base_row.get("features", {}),
                feature_list=base_row.get("feature_list", []),
                seller_stats=stats_map.get(candidate.random_key, SellerStats()),
                top_offers=offers_map.get(candidate.random_key, []),
            )
        )
        emitted.add(candidate.random_key)
    return contexts


def gather_product_contexts(lookup_args: ProductLookupArgs) -> List[ProductContext]:
    """Collect and hydrate product contexts based on lookup arguments."""

    limit = max(1, min(lookup_args.limit, 20))
    candidates: List[ProductCandidate] = []
    seen: set[str] = set()

    if lookup_args.base_random_key:
        direct = _fetch_product_by_random_key(lookup_args.base_random_key.strip())
        if direct and direct.random_key not in seen:
            candidates.append(direct)
            seen.add(direct.random_key)

    if lookup_args.member_random_key:
        member = _fetch_product_by_member_key(lookup_args.member_random_key.strip())
        if member and member.random_key not in seen:
            candidates.append(member)
            seen.add(member.random_key)

    query = lookup_args.product_name.strip() if lookup_args.product_name else None
    if query:
        for match in _hybrid_retrieve(query, limit * 2):
            if len(candidates) >= limit:
                break
            if match.random_key in seen:
                continue
            candidates.append(match)
            seen.add(match.random_key)

        if len(candidates) < limit:
            for match in _sql_fuzzy_match(query, limit * 2):
                if len(candidates) >= limit:
                    break
                if match.random_key in seen:
                    continue
                candidates.append(match)
                seen.add(match.random_key)

    return _hydrate_product_contexts(candidates[:limit])


__all__ = ["gather_product_contexts"]
