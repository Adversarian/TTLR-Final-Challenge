"""Database tools consumed by the multi-turn agent."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from pydantic_ai.tools import RunContext, Tool
from sqlalchemy import Boolean, Float, Integer, Text, bindparam, text
from sqlalchemy.dialects.postgresql import JSON

from ..dependencies import AgentDependencies
from .schemas import (
    SearchCandidate,
    SearchMembersDistributions,
    SearchMembersResult,
)


async def _search_members(
    ctx: RunContext[AgentDependencies],
    *,
    priority_query_tokens: Optional[List[str]] = None,
    generic_query_tokens: Optional[List[str]] = None,
    city_id: Optional[int] = None,
    category_id: Optional[int] = None,
    brand_id: Optional[int] = None,
    city_name: Optional[str] = None,
    category_name: Optional[str] = None,
    brand_name: Optional[str] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    has_warranty: Optional[bool] = None,
    shop_min_score: Optional[float] = None,
    limit: int = 5,
) -> SearchMembersResult:
    if limit <= 0:
        limit = 5
    limit = min(limit, 10)

    priority_tokens = [
        token.strip()
        for token in (priority_query_tokens or [])
        if token and token.strip()
    ]
    generic_tokens = [
        token.strip()
        for token in (generic_query_tokens or [])
        if token and token.strip()
    ]

    priority_query_text = " ".join(priority_tokens)
    generic_query_text = " ".join(generic_tokens)
    has_priority_query = bool(priority_tokens)
    has_generic_query = bool(generic_tokens)

    def _quote(tokens: List[str]) -> List[str]:
        quoted: List[str] = []
        for token in tokens:
            cleaned = token.replace('"', " ")
            if cleaned:
                quoted.append(f'"{cleaned}"')
        return quoted

    priority_quoted_tokens = _quote(priority_tokens)
    generic_quoted_tokens = _quote(generic_tokens)
    priority_any_query_text = " OR ".join(priority_quoted_tokens)
    generic_any_query_text = " OR ".join(generic_quoted_tokens)
    has_priority_any_query = bool(priority_quoted_tokens)
    has_generic_any_query = bool(generic_quoted_tokens)

    priority_weight = 2.0 if has_priority_query else 0.0
    generic_weight = 1.0 if has_generic_query else 0.0

    brand_name_query = brand_name.strip() if brand_name and brand_name.strip() else None
    category_name_query = (
        category_name.strip() if category_name and category_name.strip() else None
    )
    city_name_query = city_name.strip() if city_name and city_name.strip() else None

    stmt = text(
        """
        WITH city_candidate AS (
            SELECT id
            FROM cities
            WHERE :city_name_query IS NOT NULL
              AND LOWER(name) = LOWER(:city_name_query)
            LIMIT 1
        ),
        filtered AS (
            SELECT
                m.random_key AS member_random_key,
                bp.persian_name AS base_name,
                COALESCE(br.title, 'بدون برند') AS brand_name,
                bp.brand_id,
                bp.category_id,
                m.price,
                s.id AS shop_id,
                s.score AS shop_score,
                s.city_id,
                city.name AS city_name,
                s.has_warranty,
                (
                    CASE
                        WHEN (:priority_weight + :generic_weight) > 0
                            THEN (
                                (
                                    0.105
                                        * (
                                            :priority_weight
                                                * (
                                                    CASE
                                                        WHEN :has_priority_query
                                                            THEN ts_rank_cd(
                                                                bp.search_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :priority_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                            + :generic_weight
                                                * (
                                                    CASE
                                                        WHEN :has_generic_query
                                                            THEN ts_rank_cd(
                                                                bp.search_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :generic_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                        )
                                    + 0.195
                                        * (
                                            :priority_weight
                                                * (
                                                    CASE
                                                        WHEN :has_priority_query
                                                            THEN ts_rank_cd(
                                                                bp.extra_features_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :priority_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                            + :generic_weight
                                                * (
                                                    CASE
                                                        WHEN :has_generic_query
                                                            THEN ts_rank_cd(
                                                                bp.extra_features_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :generic_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                        )
                                    + 0.35
                                        * (
                                            :priority_weight
                                                * COALESCE(priority_token_stats.max_phrase_rank, 0.0)
                                            + :generic_weight
                                                * COALESCE(generic_token_stats.max_phrase_rank, 0.0)
                                        )
                                    + 0.25
                                        * (
                                            :priority_weight
                                                * (
                                                    CASE
                                                        WHEN :has_priority_any_query
                                                            THEN ts_rank_cd(
                                                                bp.search_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :priority_any_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                            + :generic_weight
                                                * (
                                                    CASE
                                                        WHEN :has_generic_any_query
                                                            THEN ts_rank_cd(
                                                                bp.search_vector,
                                                                websearch_to_tsquery(
                                                                    'simple',
                                                                    :generic_any_query_text
                                                                )
                                                            )
                                                        ELSE 0.0
                                                    END
                                                )
                                        )
                                    + 0.1
                                        * (
                                            :priority_weight
                                                * COALESCE(priority_token_stats.max_similarity, 0.0)
                                            + :generic_weight
                                                * COALESCE(generic_token_stats.max_similarity, 0.0)
                                        )
                                ) / NULLIF(:priority_weight + :generic_weight, 0)
                            )
                        ELSE 0.0
                    END
                    + CASE
                        WHEN :brand_name_query IS NOT NULL
                            THEN 0.2 * similarity(COALESCE(br.title, ''), :brand_name_query)
                        ELSE 0.0
                    END
                    + CASE
                        WHEN :category_name_query IS NOT NULL
                            THEN 0.2 * similarity(COALESCE(cat.title, ''), :category_name_query)
                        ELSE 0.0
                    END
                    + CASE
                        WHEN :city_name_query IS NOT NULL
                             AND COALESCE(:city_id, city_match.id) IS NULL
                            THEN 0.1 * similarity(COALESCE(city.name, ''), :city_name_query)
                        ELSE 0.0
                    END
                ) AS relevance
            FROM members AS m
            JOIN base_products AS bp ON bp.random_key = m.base_random_key
            LEFT JOIN brands AS br ON br.id = bp.brand_id
            LEFT JOIN categories AS cat ON cat.id = bp.category_id
            JOIN shops AS s ON s.id = m.shop_id
            LEFT JOIN cities AS city ON city.id = s.city_id
            LEFT JOIN city_candidate AS city_match ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                    MAX(
                        ts_rank_cd(
                            bp.search_vector,
                            websearch_to_tsquery(
                                'simple',
                                CONCAT('"', replace(token_text, '"', ' '), '"')
                            )
                        )
                    ) AS max_phrase_rank,
                    MAX(
                        GREATEST(
                            similarity(bp.persian_name, token_text),
                            similarity(COALESCE(bp.english_name, ''), token_text)
                        )
                    ) AS max_similarity
                FROM json_array_elements_text(:priority_tokens_json) AS tokens(token_text)
            ) AS priority_token_stats ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                    MAX(
                        ts_rank_cd(
                            bp.search_vector,
                            websearch_to_tsquery(
                                'simple',
                                CONCAT('"', replace(token_text, '"', ' '), '"')
                            )
                        )
                    ) AS max_phrase_rank,
                    MAX(
                        GREATEST(
                            similarity(bp.persian_name, token_text),
                            similarity(COALESCE(bp.english_name, ''), token_text)
                        )
                    ) AS max_similarity
                FROM json_array_elements_text(:generic_tokens_json) AS tokens(token_text)
            ) AS generic_token_stats ON TRUE
            WHERE (:brand_id IS NULL OR bp.brand_id = :brand_id)
              AND (:category_id IS NULL OR bp.category_id = :category_id)
              AND (
                    COALESCE(:city_id, city_match.id) IS NULL
                    OR s.city_id = COALESCE(:city_id, city_match.id)
                )
              AND (:price_min IS NULL OR m.price >= :price_min)
              AND (:price_max IS NULL OR m.price <= :price_max)
              AND (:has_warranty IS NULL OR s.has_warranty = :has_warranty)
              AND (:shop_min_score IS NULL OR s.score >= :shop_min_score)
        ),
        ranked AS (
            SELECT
                filtered.*,
                ROW_NUMBER() OVER (
                    ORDER BY filtered.relevance DESC NULLS LAST,
                             CASE
                                 WHEN :price_min IS NULL AND :price_max IS NULL
                                     THEN filtered.price
                             END ASC NULLS LAST,
                             CASE
                                 WHEN :shop_min_score IS NULL THEN filtered.shop_score
                             END DESC NULLS LAST,
                             filtered.price ASC,
                             filtered.shop_score DESC,
                             filtered.member_random_key
                ) AS row_number
            FROM filtered
        ),
        brand_distribution AS (
            SELECT brand_id, COUNT(*) AS freq
            FROM filtered
            GROUP BY brand_id
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ),
        city_distribution AS (
            SELECT city_id, COUNT(*) AS freq
            FROM filtered
            GROUP BY city_id
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ),
        warranty_distribution AS (
            SELECT has_warranty, COUNT(*) AS freq
            FROM filtered
            GROUP BY has_warranty
        ),
        price_bounds AS (
            SELECT MIN(price) AS min_price, MAX(price) AS max_price FROM filtered
        ),
        price_series AS (
            SELECT
                CASE
                    WHEN bounds.max_price IS NULL THEN 'نامشخص'
                    WHEN bounds.max_price = bounds.min_price THEN to_char(bounds.max_price, 'FM9999999999')
                    ELSE (
                        CASE buckets.bucket
                            WHEN 1 THEN CONCAT('≤ ', to_char(bounds.min_price + buckets.step, 'FM9999999999'))
                            WHEN 4 THEN CONCAT('≥ ', to_char(bounds.max_price - buckets.step, 'FM9999999999'))
                            ELSE CONCAT(
                                to_char(bounds.min_price + (buckets.bucket - 1) * buckets.step, 'FM9999999999'),
                                '–',
                                to_char(bounds.min_price + buckets.bucket * buckets.step, 'FM9999999999')
                            )
                        END
                    )
                END AS price_band
            FROM filtered AS f
            CROSS JOIN price_bounds AS bounds
            CROSS JOIN LATERAL (
                SELECT
                    CASE
                        WHEN bounds.max_price IS NULL OR bounds.min_price IS NULL THEN NULL
                        WHEN bounds.max_price = bounds.min_price THEN 1
                        ELSE width_bucket(f.price, bounds.min_price, bounds.max_price, 4)
                    END AS bucket,
                    CASE
                        WHEN bounds.max_price IS NULL OR bounds.min_price IS NULL THEN 0
                        WHEN bounds.max_price = bounds.min_price THEN 0
                        ELSE GREATEST((bounds.max_price - bounds.min_price) / 4.0, 1)
                    END AS step
            ) AS buckets
        ),
        price_distribution AS (
            SELECT price_band, COUNT(*) AS freq
            FROM price_series
            GROUP BY price_band
            ORDER BY COUNT(*) DESC
            LIMIT 10
        )
        SELECT json_build_object(
            'count', (SELECT COUNT(*) FROM filtered),
            'topK', COALESCE(
                (
                    SELECT json_agg(
                        json_build_object(
                            'member_random_key', member_random_key,
                            'base_name',        base_name,
                            'brand',            brand_name,
                            'price',            price,
                            'shop_name',        CONCAT('فروشگاه ', shop_id::text),
                            'shop_score',       shop_score,
                            'city_name',        city_name,
                            'relevance',        relevance
                        )
                        ORDER BY relevance DESC NULLS LAST,
                                 CASE
                                     WHEN :price_min IS NULL AND :price_max IS NULL
                                         THEN price
                                 END ASC NULLS LAST,
                                 CASE
                                     WHEN :shop_min_score IS NULL THEN shop_score
                                 END DESC NULLS LAST,
                                 price ASC,
                                 shop_score DESC,
                                 member_random_key
                    )
                    FROM ranked
                    WHERE row_number <= :limit
                      AND relevance > 0
                ),
                '[]'::json
            ),
            'distributions', json_build_object(
                'brand', (
                    SELECT COALESCE(json_agg(json_build_array(brand_id, freq)), '[]'::json)
                    FROM brand_distribution
                ),
                'city', (
                    SELECT COALESCE(json_agg(json_build_array(city_id, freq)), '[]'::json)
                    FROM city_distribution
                ),
                'price_band', (
                    SELECT COALESCE(json_agg(json_build_array(price_band, freq)), '[]'::json)
                    FROM price_distribution
                ),
                'warranty', (
                    SELECT COALESCE(json_agg(json_build_array(has_warranty, freq)), '[]'::json)
                    FROM warranty_distribution
                )
            )
        ) AS payload;
        """
    ).bindparams(
        bindparam("priority_query_text", type_=Text()),
        bindparam("generic_query_text", type_=Text()),
        bindparam("priority_any_query_text", type_=Text()),
        bindparam("generic_any_query_text", type_=Text()),
        bindparam("has_priority_query", type_=Boolean),
        bindparam("has_generic_query", type_=Boolean),
        bindparam("has_priority_any_query", type_=Boolean),
        bindparam("has_generic_any_query", type_=Boolean),
        bindparam("priority_tokens_json", type_=JSON),
        bindparam("generic_tokens_json", type_=JSON),
        bindparam("priority_weight", type_=Float),
        bindparam("generic_weight", type_=Float),
        bindparam("brand_name_query", type_=Text()),
        bindparam("category_name_query", type_=Text()),
        bindparam("city_name_query", type_=Text()),
        bindparam("brand_id", type_=Integer),
        bindparam("category_id", type_=Integer),
        bindparam("city_id", type_=Integer),
        bindparam("price_min", type_=Integer),
        bindparam("price_max", type_=Integer),
        bindparam("has_warranty", type_=Boolean),
        bindparam("shop_min_score", type_=Float),
        bindparam("limit", type_=Integer),
    )

    params = {
        "priority_query_text": priority_query_text,
        "generic_query_text": generic_query_text,
        "priority_any_query_text": priority_any_query_text,
        "generic_any_query_text": generic_any_query_text,
        "has_priority_query": has_priority_query,
        "has_generic_query": has_generic_query,
        "has_priority_any_query": has_priority_any_query,
        "has_generic_any_query": has_generic_any_query,
        "priority_tokens_json": priority_tokens,
        "generic_tokens_json": generic_tokens,
        "priority_weight": priority_weight,
        "generic_weight": generic_weight,
        "brand_name_query": brand_name_query,
        "category_name_query": category_name_query,
        "city_name_query": city_name_query,
        "brand_id": brand_id,
        "category_id": category_id,
        "city_id": city_id,
        "price_min": price_min,
        "price_max": price_max,
        "has_warranty": has_warranty,
        "shop_min_score": shop_min_score,
        "limit": limit,
    }

    session = ctx.deps.session
    result = await session.execute(stmt, params)
    row = result.one()
    payload_value = row._mapping.get("payload") if hasattr(row, "_mapping") else row[0]
    if payload_value is None:
        data = {"count": 0, "topK": [], "distributions": {}}
    elif isinstance(payload_value, str):
        data = json.loads(payload_value)
    else:
        data = dict(payload_value)

    top_candidates = [
        SearchCandidate(**candidate) for candidate in data.get("topK", [])
    ]
    distributions = data.get("distributions", {}) or {}

    def _coerce_sequence(key: str) -> Optional[List[Tuple[object, int]]]:
        value = distributions.get(key)
        if value is None:
            return None
        out: List[Tuple[object, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    out.append((item[0], int(item[1])))
                except (TypeError, ValueError):
                    continue
        return out or None

    return SearchMembersResult(
        count=int(data.get("count", 0) or 0),
        topK=top_candidates,
        distributions=SearchMembersDistributions(
            brand=_coerce_sequence("brand"),
            city=_coerce_sequence("city"),
            price_band=_coerce_sequence("price_band"),
            warranty=_coerce_sequence("warranty"),
        ),
    )


SEARCH_MEMBERS_TOOL = Tool(
    _search_members,
    name="search_members",
    description=(
        "Search Member (BaseProduct × Shop) candidates using the hard filters and "
        "soft relevance scoring over product names and features. Call this tool to "
        "filter products based on the user's requirements only after asking the "
        "necessary clarification questions, and never invoke it on the first turn of "
        "a conversation."
    ),
)


__all__ = ["SEARCH_MEMBERS_TOOL"]
