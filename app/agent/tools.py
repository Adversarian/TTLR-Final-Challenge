"""Tool implementations exposed to the agent."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic_ai import Tool
from pydantic_ai.tools import RunContext

from ..services.feature_utils import (
    convert_value,
    extract_value,
    find_attribute_key,
    normalize_unit,
)
from ..services.preprocess import normalize_query
from ..db import DatabasePool
from .context import AgentDependencies
from .models import FeatureLookupResult, ProductResolveResult, SellerStatsResult

PRODUCT_RESOLVE_SQL = """
SELECT
    random_key AS rk,
    0.6 * ts_rank_cd(to_tsvector('simple', search_text), plainto_tsquery('simple', $1))
    + 0.4 * similarity(search_text, $1) AS score
FROM base_products
WHERE ($2::text IS NULL OR category_path ILIKE '%' || $2 || '%')
  AND ($3::text IS NULL OR brand_title ILIKE '%' || $3 || '%')
ORDER BY score DESC
LIMIT $4
"""

BASE_FEATURE_SQL = """
SELECT extra_features
FROM base_products
WHERE random_key = $1
"""

SELLER_STATS_SQL = """
WITH m AS (
    SELECT m.random_key AS member_rk,
           m.price,
           s.id AS shop_id,
           s.score,
           s.has_warranty,
           s.city_id
    FROM members m
    JOIN shops s ON s.id = m.shop_id
    WHERE m.base_random_key = $1
      AND ($2::boolean IS NULL OR s.has_warranty = $2)
      AND ($3::numeric IS NULL OR s.score >= $3)
      AND ($4::integer IS NULL OR s.city_id = $4)
)
SELECT
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    AVG(price) AS avg_price,
    COUNT(*) AS cnt,
    (SELECT shop_id FROM m ORDER BY score DESC, price ASC LIMIT 1) AS top_shop
FROM m
"""


async def _run_product_resolve(
    database: DatabasePool,
    query: str,
    category_hint: Optional[str] = None,
    brand_hint: Optional[str] = None,
    limit: int = 5,
) -> ProductResolveResult:
    normalized_query = normalize_query(query)
    async with database.connection() as conn:
        rows = await conn.fetch(PRODUCT_RESOLVE_SQL, normalized_query, category_hint, brand_hint, limit)
    candidates = [
        {"rk": row["rk"], "score": float(row["score"]) if row["score"] is not None else 0.0}
        for row in rows
    ]
    base_key = candidates[0]["rk"] if candidates else None
    return ProductResolveResult(base_random_key=base_key, candidates=candidates)


async def product_resolve(
    ctx: RunContext[AgentDependencies],
    query: str,
    category_hint: Optional[str] = None,
    brand_hint: Optional[str] = None,
    limit: int = 5,
) -> ProductResolveResult:
    """Map user text to a base random key using trigram + FTS."""

    return await _run_product_resolve(ctx.deps.database, query, category_hint, brand_hint, limit)


async def feature_lookup(
    ctx: RunContext[AgentDependencies],
    attribute_phrase: str,
    base_random_key: Optional[str] = None,
    query: Optional[str] = None,
) -> FeatureLookupResult:
    """Return the attribute value for a base product."""

    resolved_key = base_random_key
    if resolved_key is None and query:
        resolution = await _run_product_resolve(ctx.deps.database, query, None, None, limit=1)
        resolved_key = resolution.base_random_key
    if resolved_key is None:
        return FeatureLookupResult(needs_clarification=True)

    async with ctx.deps.database.connection() as conn:
        row = await conn.fetchrow(BASE_FEATURE_SQL, resolved_key)
    if row is None:
        return FeatureLookupResult(needs_clarification=True)

    features: Dict[str, Any] = row["extra_features"] or {}
    key = find_attribute_key(attribute_phrase, features.keys())
    if key is None:
        return FeatureLookupResult(value_text=None, provenance_key=resolved_key)

    value = features.get(key)
    numeric_value, unit, text = extract_value(value)
    standard_unit = normalize_unit(unit)
    display_value = text
    if numeric_value is not None:
        target_unit = standard_unit
        normalized_value, normalized_unit = convert_value(numeric_value, standard_unit, target_unit)
        if normalized_unit:
            display_value = f"{normalized_value:g} {normalized_unit}"
        else:
            display_value = f"{normalized_value:g}"
    return FeatureLookupResult(
        value_text=display_value,
        raw_value=numeric_value,
        unit=standard_unit,
        provenance_key=resolved_key,
    )


async def seller_stats(
    ctx: RunContext[AgentDependencies],
    metric: str,
    base_random_key: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> SellerStatsResult:
    """Aggregate seller metrics for a base product."""

    metric_key = metric.lower()
    if metric_key not in {"min", "max", "avg", "count", "top_shop"}:
        raise ValueError("Unsupported metric")

    resolved_key = base_random_key
    if resolved_key is None and query:
        resolution = await _run_product_resolve(ctx.deps.database, query, None, None, limit=1)
        resolved_key = resolution.base_random_key
    if resolved_key is None:
        return SellerStatsResult(result=None)

    filters = filters or {}
    has_warranty = filters.get("has_warranty")
    score_min = filters.get("score_min")
    city_id = filters.get("city_id")

    async with ctx.deps.database.connection() as conn:
        row = await conn.fetchrow(
            SELLER_STATS_SQL,
            resolved_key,
            has_warranty,
            score_min,
            city_id,
        )
    if row is None:
        return SellerStatsResult(result=None)

    if metric_key == "min":
        value = row["min_price"]
    elif metric_key == "max":
        value = row["max_price"]
    elif metric_key == "avg":
        value = row["avg_price"]
    elif metric_key == "count":
        value = row["cnt"]
    else:
        value = row["top_shop"]

    result_text: Optional[str]
    shop_id: Optional[int] = None
    if metric_key == "top_shop":
        result_text = str(value) if value is not None else None
        shop_id = value if isinstance(value, int) else None
    else:
        if value is None:
            result_text = None
        else:
            if isinstance(value, float):
                result_text = f"{value:.2f}".rstrip("0").rstrip(".")
            else:
                result_text = str(value)
    return SellerStatsResult(result=result_text, shop_id=shop_id)


PRODUCT_RESOLVE_TOOL = Tool(product_resolve, name="ProductResolve", description="Resolve user text to a base_random_key")
FEATURE_LOOKUP_TOOL = Tool(feature_lookup, name="FeatureLookup", description="Lookup a feature value for a base product")
SELLER_STATS_TOOL = Tool(seller_stats, name="SellerStats", description="Compute seller metrics for a base product")

TOOLKIT = [PRODUCT_RESOLVE_TOOL, FEATURE_LOOKUP_TOOL, SELLER_STATS_TOOL]
"""Collection of tools made available to the agent."""
