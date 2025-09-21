"""Tool definitions and helper functions for the shopping assistant."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import List, Sequence

from pydantic_ai.tools import RunContext, Tool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import BaseProduct, City, Member, Shop
from .dependencies import AgentDependencies
from .schemas import (
    CitySellerStatistics,
    FeatureLookupResult,
    ProductFeature,
    ProductMatch,
    ProductSearchResult,
    SellerStatistics,
)


def _normalize_text(value: str) -> str:
    """Return a lightly normalised version of Persian/English text."""

    replacements = {
        "\u064a": "ی",  # Arabic Yeh -> Persian Yeh
        "\u0643": "ک",  # Arabic Kaf -> Persian Keheh
        "\u06cc": "ی",  # Farsi Yeh variant
        "\u06a9": "ک",  # Keheh variant
    }
    lowered = value.strip().lower()
    for src, dest in replacements.items():
        lowered = lowered.replace(src, dest)
    return lowered


async def _fetch_top_matches(
    session: AsyncSession, normalized_query: str, limit: int = 10
) -> Sequence[ProductMatch]:
    """Return the strongest matches for the provided search query."""

    if not normalized_query:
        return []

    similarity_name = func.greatest(
        func.similarity(BaseProduct.persian_name, normalized_query),
        func.similarity(func.coalesce(BaseProduct.english_name, ""), normalized_query),
    )

    score = similarity_name.label("score")

    stmt = (
        select(
            BaseProduct.random_key,
            BaseProduct.persian_name,
            BaseProduct.english_name,
            score,
        )
        .order_by(score.desc())
        .where(similarity_name > 0.0)
        .limit(limit)
    )

    result = await session.execute(stmt)
    matches: List[ProductMatch] = []
    for random_key, persian_name, english_name, score in result:
        matches.append(
            ProductMatch(
                random_key=random_key,
                persian_name=persian_name,
                english_name=english_name,
                similarity=float(score or 0.0),
            )
        )
    return matches


async def _find_product_by_key(
    session: AsyncSession, raw_query: str
) -> ProductMatch | None:
    """Return a direct random-key match when one exists."""

    trimmed = raw_query.strip()
    if not trimmed:
        return None

    product = await session.get(BaseProduct, trimmed)
    if product is None:
        return None

    return ProductMatch(
        random_key=product.random_key,
        persian_name=product.persian_name,
        english_name=product.english_name,
        similarity=1.0,
    )


def _flatten_features(extra_features: dict | None) -> List[tuple[str, str]]:
    """Flatten a nested JSON blob into simple feature/value pairs."""

    if not extra_features:
        return []

    flattened: List[tuple[str, str]] = []

    def _walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                new_prefix = f"{prefix} {key}".strip()
                _walk(new_prefix, nested)
        elif isinstance(value, list):
            str_value = ", ".join(str(item) for item in value)
            flattened.append((prefix, str_value))
        else:
            flattened.append((prefix, str(value)))

    for key, value in extra_features.items():
        _walk(key, value)

    return flattened


async def _search_base_products(
    ctx: RunContext[AgentDependencies], query: str
) -> ProductSearchResult:
    """Resolve a customer request to likely base products."""

    normalized = _normalize_text(query)
    async with ctx.deps.session_factory() as session:
        direct_match = await _find_product_by_key(session, query)
        if direct_match is not None:
            return ProductSearchResult(query=normalized, matches=[direct_match])

        matches = await _fetch_top_matches(session, normalized)
    return ProductSearchResult(query=normalized, matches=list(matches))


async def _fetch_feature_details(
    ctx: RunContext[AgentDependencies],
    base_random_key: str,
) -> FeatureLookupResult:
    """Return the full feature/value list for the requested base product."""

    normalized_key = _normalize_text(base_random_key)
    trimmed_key = base_random_key.strip()

    async with ctx.deps.session_factory() as session:
        product = await session.get(BaseProduct, trimmed_key) if trimmed_key else None

        if product is not None:
            extra_features = product.extra_features
        else:
            stmt = select(BaseProduct.extra_features).where(
                func.lower(BaseProduct.random_key) == normalized_key
            )
            result = await session.execute(stmt)
            extra_features = result.scalar_one_or_none()

    flattened = _flatten_features(
        extra_features if isinstance(extra_features, dict) else {}
    )

    features = [ProductFeature(name=name, value=value) for name, value in flattened]
    canonical_key = trimmed_key or base_random_key

    return FeatureLookupResult(
        base_random_key=canonical_key,
        features=features,
        available_features=[feature.name for feature in features],
    )


_SELLER_STATISTICS_KEYS: List[str] = [
    "total_offers",
    "distinct_shops",
    "shops_with_warranty",
    "shops_without_warranty",
    "min_price",
    "max_price",
    "average_price",
    "min_score",
    "max_score",
    "average_score",
    "num_cities_with_offers",
]

_STATISTIC_ALIASES = {
    "offer_count": "total_offers",
    "offers": "total_offers",
    "shop_count": "distinct_shops",
}

_CITY_ROLLUP_LIMIT = 50


async def _collect_seller_statistics(
    ctx: RunContext[AgentDependencies],
    base_random_key: str,
    statistic: str,
    city: str | None = None,
) -> SellerStatistics:
    """Aggregate pricing, warranty, and score data for one base product."""

    trimmed_key = base_random_key.strip()
    requested_city = city.strip() if city else None

    canonical_stat = _STATISTIC_ALIASES.get(
        statistic.strip().lower() if statistic else "", "total_offers"
    )
    if canonical_stat not in _SELLER_STATISTICS_KEYS:
        canonical_stat = "total_offers"

    async with ctx.deps.session_factory() as session:
        stmt = (
            select(
                Member.shop_id,
                Member.price,
                Shop.has_warranty,
                Shop.score,
                Shop.city_id,
                City.name,
            )
            .join(Shop, Shop.id == Member.shop_id)
            .join(City, City.id == Shop.city_id)
            .where(Member.base_random_key == trimmed_key)
        )

        result = await session.execute(stmt)
        offer_records = list(result)

    seen_shop_ids: set[int] = set()
    price_samples: List[int] = []
    score_samples: List[float] = []
    shops_with_warranty = 0

    city_buckets = defaultdict(
        lambda: {
            "city_id": None,
            "city_name": None,
            "offer_count": 0,
            "shops_with_warranty": 0,
            "shop_ids": set(),
            "prices": [],
            "scores": [],
        }
    )

    for shop_id, price, has_warranty, score, city_id, city_name in offer_records:
        seen_shop_ids.add(int(shop_id))
        price_value = int(price) if price is not None else None
        score_value = float(score) if score is not None else None

        if price_value is not None:
            price_samples.append(price_value)
        if score_value is not None:
            score_samples.append(score_value)
        if bool(has_warranty):
            shops_with_warranty += 1

        entry = city_buckets[city_id]
        entry["city_id"] = int(city_id) if city_id is not None else None
        entry["city_name"] = city_name
        entry["offer_count"] += 1
        entry["shops_with_warranty"] += 1 if bool(has_warranty) else 0
        entry["shop_ids"].add(int(shop_id))
        if price_value is not None:
            entry["prices"].append(price_value)
        if score_value is not None:
            entry["scores"].append(score_value)

    total_offers = len(offer_records)
    shops_without_warranty = total_offers - shops_with_warranty

    min_price = min(price_samples) if price_samples else None
    max_price = max(price_samples) if price_samples else None
    average_price = round(mean(price_samples), 2) if price_samples else None

    min_score = min(score_samples) if score_samples else None
    max_score = max(score_samples) if score_samples else None
    average_score = round(mean(score_samples), 2) if score_samples else None

    city_rollups: List[CitySellerStatistics] = []
    for entry in city_buckets.values():
        price_list: List[int] = entry.pop("prices")
        score_list: List[float] = entry.pop("scores")
        shop_ids: set[int] = entry.pop("shop_ids")
        offer_count = entry["offer_count"]
        with_warranty = entry["shops_with_warranty"]

        city_rollups.append(
            CitySellerStatistics(
                city_id=entry["city_id"],
                city_name=entry["city_name"],
                offer_count=offer_count,
                distinct_shops=len(shop_ids),
                shops_with_warranty=with_warranty,
                shops_without_warranty=offer_count - with_warranty,
                min_price=min(price_list) if price_list else None,
                max_price=max(price_list) if price_list else None,
                average_price=round(mean(price_list), 2) if price_list else None,
                min_score=min(score_list) if score_list else None,
                max_score=max(score_list) if score_list else None,
                average_score=round(mean(score_list), 2) if score_list else None,
            )
        )

    city_rollups.sort(key=lambda item: item.offer_count, reverse=True)
    matched_city: CitySellerStatistics | None = None
    if requested_city:
        normalized_city = requested_city.lower()
        for entry in city_rollups:
            if (entry.city_name or "").lower() == normalized_city:
                matched_city = entry
                break

    stat_baseline = {
        "total_offers": total_offers,
        "distinct_shops": len(seen_shop_ids),
        "shops_with_warranty": shops_with_warranty,
        "shops_without_warranty": shops_without_warranty,
        "min_price": min_price,
        "max_price": max_price,
        "average_price": average_price,
        "min_score": min_score,
        "max_score": max_score,
        "average_score": average_score,
        "num_cities_with_offers": len(city_rollups),
    }

    statistic_value: float | int | None = stat_baseline.get(canonical_stat)
    if matched_city and canonical_stat in {
        "total_offers",
        "distinct_shops",
        "shops_with_warranty",
        "shops_without_warranty",
        "min_price",
        "max_price",
        "average_price",
        "min_score",
        "max_score",
        "average_score",
    }:
        city_values = {
            "total_offers": matched_city.offer_count,
            "distinct_shops": matched_city.distinct_shops,
            "shops_with_warranty": matched_city.shops_with_warranty,
            "shops_without_warranty": matched_city.shops_without_warranty,
            "min_price": matched_city.min_price,
            "max_price": matched_city.max_price,
            "average_price": matched_city.average_price,
            "min_score": matched_city.min_score,
            "max_score": matched_city.max_score,
            "average_score": matched_city.average_score,
        }
        statistic_value = city_values.get(canonical_stat)

    if canonical_stat == "num_cities_with_offers":
        statistic_value = len(city_rollups)

    if requested_city and matched_city:
        city_rollup_slice = [matched_city]
    elif requested_city:
        city_rollup_slice = city_rollups[:_CITY_ROLLUP_LIMIT]
    else:
        city_rollup_slice = city_rollups[:_CITY_ROLLUP_LIMIT]

    return SellerStatistics(
        base_random_key=trimmed_key,
        statistic=canonical_stat,
        city=requested_city,
        value=statistic_value,
        total_offers=total_offers,
        distinct_shops=len(seen_shop_ids),
        shops_with_warranty=shops_with_warranty,
        shops_without_warranty=shops_without_warranty,
        min_price=min_price,
        max_price=max_price,
        average_price=average_price,
        min_score=min_score,
        max_score=max_score,
        average_score=average_score,
        num_cities_with_offers=len(city_rollups),
        available_statistics=list(_SELLER_STATISTICS_KEYS),
        city_stats=city_rollup_slice,
    )


PRODUCT_SEARCH_TOOL = Tool(
    _search_base_products,
    name="search_base_products",
    description=(
        "Use this tool to map the customer's language to actual base products. "
        "Your search string MUST be an exact substring from the latest user message. "
        "Provide that snippet from the customer request so the tool can run a fuzzy lookup. "
        "It returns up to 10 of the strongest catalogue matches (best first) with random keys and similarity scores."
    ),
)


FEATURE_LOOKUP_TOOL = Tool(
    _fetch_feature_details,
    name="get_product_feature",
    description=(
        "Use this tool after you know which base product the user means. "
        "Provide only the base product random key to receive every catalogue feature "
        "as name/value pairs along with a convenience list of feature names. "
        "Pick the attribute you need from the returned payload."
    ),
)


SELLER_STATISTICS_TOOL = Tool(
    _collect_seller_statistics,
    name="get_seller_statistics",
    description=(
        "After identifying the base product, call this to summarise seller activity. "
        "Provide the base random key, specify the statistic name you must report (for example: "
        "total_offers, min_price, average_price, max_score), and optionally pass a Persian city name "
        "to focus on that location. It returns aggregated counts, price extrema, and score summaries. "
        "Use the `value` field from the response to populate your numeric_answer and reply with digits only."
    ),
)


__all__ = [
    "PRODUCT_SEARCH_TOOL",
    "FEATURE_LOOKUP_TOOL",
    "SELLER_STATISTICS_TOOL",
    "_fetch_feature_details",
]
