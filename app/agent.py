"""Agent setup and tool definitions for the shopping assistant."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import os
from statistics import mean
from typing import List, Sequence

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext, Tool
from pydantic_ai.settings import ModelSettings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import BaseProduct, City, Member, Shop


@dataclass
class AgentDependencies:
    """Runtime dependencies passed to the agent on every run."""

    session: AsyncSession


class ProductMatch(BaseModel):
    """Represents a single product candidate returned by a search."""

    random_key: str = Field(..., description="The base product random key.")
    persian_name: str = Field(..., description="Canonical Persian name of the product.")
    english_name: str | None = Field(
        None, description="English product name when available."
    )
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score relative to the input query."
    )


class ProductSearchResult(BaseModel):
    """Collection of ranked product matches for a customer request."""

    query: str = Field(..., description="Normalized query that was searched.")
    matches: List[ProductMatch] = Field(
        default_factory=list,
        description="Ranked list of the strongest matches (best first).",
    )


class ProductFeature(BaseModel):
    """Single flattened feature/value pair for a product."""

    name: str = Field(..., description="Feature label as stored in the catalogue.")
    value: str = Field(..., description="Human readable value for the feature.")


class FeatureLookupResult(BaseModel):
    """Complete catalogue feature map for a base product."""

    base_random_key: str = Field(..., description="Target product random key.")
    features: List[ProductFeature] = Field(
        default_factory=list,
        description="All feature/value pairs extracted from the catalogue record.",
    )
    available_features: List[str] = Field(
        default_factory=list,
        description=(
            "Convenience list of feature names to help pick the relevant attribute."
        ),
    )


class CitySellerStatistics(BaseModel):
    """Aggregated seller metrics for a specific city."""

    city_id: int | None = Field(None, description="Identifier of the city.")
    city_name: str | None = Field(None, description="Display name of the city.")
    offer_count: int = Field(..., ge=0, description="Total offers from this city.")
    distinct_shops: int = Field(
        ..., ge=0, description="Number of unique shops contributing offers."
    )
    shops_with_warranty: int = Field(
        ..., ge=0, description="Offers backed by Torob warranty."
    )
    shops_without_warranty: int = Field(
        ..., ge=0, description="Offers without Torob warranty."
    )
    min_price: int | None = Field(
        None, description="Cheapest price observed in this city."
    )
    max_price: int | None = Field(
        None, description="Most expensive price observed in this city."
    )
    average_price: float | None = Field(
        None, description="Average offer price for this city."
    )
    min_score: float | None = Field(
        None, description="Lowest shop score observed in this city."
    )
    max_score: float | None = Field(
        None, description="Highest shop score observed in this city."
    )
    average_score: float | None = Field(
        None, description="Average shop score for this city."
    )


class SellerStatistics(BaseModel):
    """Aggregated marketplace data for a base product."""

    base_random_key: str = Field(..., description="Target base product key.")
    statistic: str = Field(
        ..., description="Requested statistic key that maps to the numeric value."
    )
    city: str | None = Field(
        None,
        description="Optional Persian city name used to narrow down the statistic.",
    )
    value: float | int | None = Field(
        None, description="Numeric value for the requested statistic."
    )
    total_offers: int = Field(..., ge=0, description="Total number of offers.")
    distinct_shops: int = Field(
        ..., ge=0, description="Number of unique shops listing the product."
    )
    shops_with_warranty: int = Field(
        ..., ge=0, description="Offers sold with Torob warranty."
    )
    shops_without_warranty: int = Field(
        ..., ge=0, description="Offers sold without Torob warranty."
    )
    min_price: int | None = Field(
        None, description="Cheapest price offered across all shops."
    )
    max_price: int | None = Field(
        None, description="Most expensive price offered across all shops."
    )
    average_price: float | None = Field(
        None, description="Average price across all offers."
    )
    min_score: float | None = Field(
        None, description="Lowest shop score for the product offers."
    )
    max_score: float | None = Field(
        None, description="Highest shop score for the product offers."
    )
    average_score: float | None = Field(
        None, description="Average shop score across offers."
    )
    num_cities_with_offers: int = Field(
        ..., ge=0, description="Number of cities that list the product."
    )
    available_statistics: List[str] = Field(
        default_factory=list,
        description="List of supported statistic keys for the request.",
    )
    city_stats: List[CitySellerStatistics] = Field(
        default_factory=list,
        description="Optional per-city breakdowns supporting the statistic.",
    )


class AgentReply(BaseModel):
    """Structured response emitted by the agent."""

    message: str | None = Field(
        None, description="Assistant message to display to the customer."
    )
    base_random_keys: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Base product keys relevant to the response (at most 10).",
    )
    member_random_keys: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Member product keys relevant to the response (at most 10).",
    )
    numeric_answer: Decimal | None = Field(
        None,
        description=(
            "When responding with a numeric seller statistic, populate this field "
            "with the exact value so the API layer can enforce digit-only replies."
        ),
    )

    def clipped(self) -> "AgentReply":
        """Return a copy trimmed to the API list length limits."""

        return AgentReply(
            message=self.message,
            base_random_keys=self.base_random_keys[:10],
            member_random_keys=self.member_random_keys[:10],
            numeric_answer=self.numeric_answer,
        )


def _configure_logfire() -> None:
    """Configure Logfire instrumentation if it has not been configured yet."""

    token = os.getenv("LOGFIRE_API_KEY")
    if token:
        logfire.configure(token=token)
    else:
        logfire.configure()


_LOGFIRE_READY = False


def _ensure_logfire() -> None:
    global _LOGFIRE_READY
    if not _LOGFIRE_READY:
        _configure_logfire()
        _LOGFIRE_READY = True


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
    session = ctx.deps.session

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
    session = ctx.deps.session

    trimmed_key = base_random_key.strip()
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

    session = ctx.deps.session

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

    session = ctx.deps.session

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


SYSTEM_PROMPT = (
    "You are a concise but helpful shopping assistant. Ground every answer in the product catalogue by identifying the most relevant base product before making recommendations or quoting attributes.\n\n"
    "SCENARIO GUIDE:\n"
    "- Product procurement requests: resolve the customer's wording to a single catalogue item and answer in one turn with the best-matching base random key.\n"
    "- Feature clarification requests: locate the product first, pull the complete feature list with get_product_feature, then surface the requested attribute's value directly without asking for more details.\n"
    "- Seller competition or pricing metrics: once the product is known, fetch the required statistic with get_seller_statistics and report only the numeric result.\n"
    "- Connectivity pings or other simple sanity checks may be answered with the static shortcuts provided by the API layer.\n\n"
    "GENERAL PRINCIPLES:\n"
    "- Answer deterministic product or feature questions in a single turn; do not ask clarifying questions even if confidence is modest—pick the strongest match and, if necessary, acknowledge uncertainty succinctly.\n"
    "- Always ground statements in actual catalogue data and keep explanations brief, factual, and free of invented information.\n"
    "- When a numeric seller statistic is requested, set numeric_answer to the value from get_seller_statistics and ensure the final message contains digits only with no additional text.\n"
    "- Only include product keys when they are explicitly required or necessary for the response, keeping lists trimmed to at most one base key by default.\n\n"
    "TOOL USAGE:\n"
    "- search_base_products: Call this whenever you need to resolve what product the user references. The search string MUST be an exact substring of the user's latest message. Review up to ten returned matches and choose the option whose identifiers appear verbatim in the request.\n"
    "- get_product_feature: After identifying the product, use this to retrieve catalogue features. Provide the base random key to receive the complete list of feature/value pairs and choose the attribute that answers the question.\n"
    "- get_seller_statistics: Invoke this once you know the base product and the user needs pricing, availability, or rating aggregates. Supply the product key, pick one supported statistic name, optionally add a Persian city name, then echo the returned value as your entire reply.\n"
)


@lru_cache(maxsize=1)
def get_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a configured agent instance with tool and logging support."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        settings=ModelSettings(temperature=0.1),
    )

    return Agent(
        model=model,
        output_type=AgentReply,
        instructions=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[PRODUCT_SEARCH_TOOL, FEATURE_LOOKUP_TOOL, SELLER_STATISTICS_TOOL],
        instrument=InstrumentationSettings(),
        name="shopping-assistant",
    )
