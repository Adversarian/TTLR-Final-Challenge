"""Specialised tools supporting the scenario 4 multi-turn workflow."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from pydantic_ai.tools import RunContext, Tool
from sqlalchemy import and_, func, or_, select

from ...models import BaseProduct, Brand, Category, City, Member, Shop
from ..dependencies import AgentDependencies
from ..tools import _flatten_features
from .schemas import (
    CategoryFeatureStatistic,
    FeatureConstraintModel,
    FilteredProduct,
    MemberFilterResponse,
    MemberOffer,
    ProductFilterResponse,
)


def _normalise_text(value: str) -> str:
    """Lightweight normalisation shared across filtering utilities."""

    replacements = {
        "\u064a": "ی",
        "\u0643": "ک",
        "\u06cc": "ی",
        "\u06a9": "ک",
    }
    lowered = value.strip().lower()
    for src, dest in replacements.items():
        lowered = lowered.replace(src, dest)
    return lowered


def _resolve_category_filter(category_hint: str | None):
    """Return SQL filter expressions that approximate the category hint."""

    if not category_hint:
        return None

    tokens = [_normalise_text(part) for part in category_hint.split() if part.strip()]
    if not tokens:
        return None

    expressions = []
    for token in tokens:
        like_pattern = f"%{token}%"
        expressions.append(func.lower(Category.title).like(like_pattern))
    return and_(*expressions) if expressions else None


def _prepare_feature_requirements(
    requirements: Iterable[FeatureConstraintModel] | None,
) -> list[tuple[str, str, str]]:
    """Convert feature requirements into normalised tuples."""

    prepared: list[tuple[str, str, str]] = []
    if not requirements:
        return prepared

    for feature in requirements:
        name = feature.name.strip()
        value = feature.value.strip()
        match = feature.match.strip().lower()
        if not name or not value:
            continue
        prepared.append((_normalise_text(name), value, match))
    return prepared


def _match_feature(
    flattened: list[tuple[str, str]],
    required_name: str,
    required_value: str,
    match: str,
) -> tuple[bool, str | None]:
    """Return whether a feature constraint is satisfied and an optional note."""

    value_note: str | None = None
    required_value_norm = _normalise_text(required_value)
    numeric_value = None
    try:
        numeric_value = float(required_value.replace(",", ""))
    except ValueError:
        numeric_value = None

    for name, value in flattened:
        normalised_name = _normalise_text(name)
        if required_name not in normalised_name and normalised_name not in required_name:
            continue

        candidate_value_norm = _normalise_text(value)
        if match == "equals" and candidate_value_norm == required_value_norm:
            return True, f"{name}: {value}"
        if match == "contains" and required_value_norm in candidate_value_norm:
            return True, f"{name}: {value}"
        if match in {"min_value", "max_value"} and numeric_value is not None:
            detected_number = _extract_first_number(candidate_value_norm)
            if detected_number is None:
                continue
            if match == "min_value" and detected_number >= numeric_value:
                return True, f"{name}: {value}"
            if match == "max_value" and detected_number <= numeric_value:
                return True, f"{name}: {value}"

    return False, value_note


def _extract_first_number(value: str) -> float | None:
    """Best-effort numeric parser that extracts the first float from text."""

    digits = []
    decimal_seen = False
    for char in value:
        if char.isdigit():
            digits.append(char)
            continue
        if char in {".", ","} and not decimal_seen:
            digits.append(".")
            decimal_seen = True
            continue
        if digits:
            break

    if not digits:
        return None

    try:
        return float("".join(digits))
    except ValueError:
        return None


async def _category_feature_statistics(
    ctx: RunContext[AgentDependencies],
    *,
    category_hint: str | None = None,
    base_random_keys: list[str] | None = None,
    sample_limit: int = 250,
    limit: int = 20,
) -> list[CategoryFeatureStatistic]:
    """Return a summary of common features for the requested scope."""

    async with ctx.deps.session_factory() as session:
        stmt = select(BaseProduct.extra_features, BaseProduct.random_key)

        if base_random_keys:
            stmt = stmt.where(BaseProduct.random_key.in_(base_random_keys))
        elif category_hint:
            category_filter = _resolve_category_filter(category_hint)
            if category_filter is not None:
                stmt = stmt.join(Category, Category.id == BaseProduct.category_id)
                stmt = stmt.where(category_filter)

        stmt = stmt.limit(sample_limit)
        result = await session.execute(stmt)
        rows = list(result)

    counter: dict[str, int] = defaultdict(int)
    samples: dict[str, set[str]] = defaultdict(set)

    for extra_features, _random_key in rows:
        flattened = _flatten_features(extra_features if isinstance(extra_features, dict) else {})
        for path, value in flattened:
            key = path.strip()
            if not key:
                continue
            counter[key] += 1
            if value:
                if len(samples[key]) < 5:
                    samples[key].add(str(value))

    ranked = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:limit]
    statistics: list[CategoryFeatureStatistic] = []
    for path, occurrences in ranked:
        statistics.append(
            CategoryFeatureStatistic(
                feature_path=path,
                sample_values=sorted(samples[path])[:5],
                occurrences=occurrences,
            )
        )

    return statistics


async def _filter_base_products_by_constraints(
    ctx: RunContext[AgentDependencies],
    *,
    category_hint: str | None = None,
    brand_preferences: list[str] | None = None,
    keywords: list[str] | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
    required_features: list[FeatureConstraintModel] | None = None,
    optional_features: list[FeatureConstraintModel] | None = None,
    excluded_features: list[FeatureConstraintModel] | None = None,
    limit: int = 20,
) -> ProductFilterResponse:
    """Search base products using structured preferences."""

    async with ctx.deps.session_factory() as session:
        price_summary = (
            select(
                Member.base_random_key.label("base_random_key"),
                func.min(Member.price).label("min_price"),
                func.max(Member.price).label("max_price"),
            )
            .group_by(Member.base_random_key)
            .subquery()
        )

        stmt = (
            select(
                BaseProduct.random_key,
                BaseProduct.persian_name,
                BaseProduct.english_name,
                Category.title,
                Brand.title,
                price_summary.c.min_price,
                price_summary.c.max_price,
                BaseProduct.extra_features,
            )
            .join(Category, Category.id == BaseProduct.category_id, isouter=True)
            .join(Brand, Brand.id == BaseProduct.brand_id, isouter=True)
            .join(price_summary, price_summary.c.base_random_key == BaseProduct.random_key, isouter=True)
        )

        category_filter = _resolve_category_filter(category_hint)
        if category_filter is not None:
            stmt = stmt.where(category_filter)

        if brand_preferences:
            terms = [term.strip() for term in brand_preferences if term.strip()]
            if terms:
                brand_conditions = [
                    func.lower(Brand.title).like(f"%{_normalise_text(term)}%") for term in terms
                ]
                stmt = stmt.where(or_(*brand_conditions))

        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                stripped = keyword.strip()
                if not stripped:
                    continue
                pattern = f"%{_normalise_text(stripped)}%"
                keyword_conditions.append(func.lower(BaseProduct.persian_name).like(pattern))
                keyword_conditions.append(func.lower(func.coalesce(BaseProduct.english_name, "")).like(pattern))
            if keyword_conditions:
                stmt = stmt.where(or_(*keyword_conditions))

        if price_min is not None:
            stmt = stmt.where(price_summary.c.max_price >= price_min)
        if price_max is not None:
            stmt = stmt.where(price_summary.c.min_price <= price_max)

        stmt = stmt.limit(limit * 3)
        result = await session.execute(stmt)
        rows = list(result)

    prepared_required = _prepare_feature_requirements(required_features)
    prepared_optional = _prepare_feature_requirements(optional_features)
    prepared_excluded = _prepare_feature_requirements(excluded_features)

    candidates: list[FilteredProduct] = []

    for (
        random_key,
        persian_name,
        english_name,
        category_title,
        brand_title,
        min_price,
        max_price,
        extra_features,
    ) in rows:
        flattened = _flatten_features(extra_features if isinstance(extra_features, dict) else {})

        # Enforce required features.
        matched_notes: list[str] = []
        required_matches = 0
        for name, value, match in prepared_required:
            ok, note = _match_feature(flattened, name, value, match)
            if not ok:
                break
            required_matches += 1
            if note:
                matched_notes.append(note)
        else:
            # Only executed when all required features passed.
            excluded_hit = False
            for name, value, match in prepared_excluded:
                present, _ = _match_feature(flattened, name, value, match)
                if present:
                    excluded_hit = True
                    break

            if excluded_hit:
                continue

            optional_matches = 0
            for name, value, match in prepared_optional:
                ok, note = _match_feature(flattened, name, value, match)
                if ok:
                    optional_matches += 1
                    if note:
                        matched_notes.append(note)

            coverage = 0.0
            if prepared_required:
                coverage += 0.7 * (required_matches / max(len(prepared_required), 1))
            else:
                coverage += 0.4
            if prepared_optional:
                coverage += 0.3 * (optional_matches / len(prepared_optional))

            candidates.append(
                FilteredProduct(
                    base_random_key=random_key,
                    persian_name=persian_name,
                    english_name=english_name,
                    category_title=category_title,
                    brand_title=brand_title,
                    min_price=int(min_price) if min_price is not None else None,
                    max_price=int(max_price) if max_price is not None else None,
                    matched_features=matched_notes,
                    match_score=min(round(coverage, 3), 1.0),
                )
            )

    candidates.sort(key=lambda product: (product.match_score, -(product.min_price or 0)), reverse=True)
    return ProductFilterResponse(candidates=candidates[:limit])


async def _filter_members_by_constraints(
    ctx: RunContext[AgentDependencies],
    *,
    base_random_key: str,
    price_min: int | None = None,
    price_max: int | None = None,
    require_warranty: bool | None = None,
    min_shop_score: float | None = None,
    city: str | None = None,
    limit: int = 10,
) -> MemberFilterResponse:
    """Return member offers for a resolved base product."""

    trimmed_key = base_random_key.strip()
    if not trimmed_key:
        return MemberFilterResponse(offers=[])

    async with ctx.deps.session_factory() as session:
        stmt = (
            select(
                Member.random_key,
                Member.price,
                Shop.id,
                Shop.has_warranty,
                Shop.score,
                City.name,
            )
            .join(Shop, Shop.id == Member.shop_id)
            .join(City, City.id == Shop.city_id, isouter=True)
            .where(Member.base_random_key == trimmed_key)
        )

        if price_min is not None:
            stmt = stmt.where(Member.price >= price_min)
        if price_max is not None:
            stmt = stmt.where(Member.price <= price_max)
        if require_warranty is True:
            stmt = stmt.where(Shop.has_warranty.is_(True))
        if min_shop_score is not None:
            stmt = stmt.where(Shop.score >= float(min_shop_score))
        if city:
            stmt = stmt.where(func.lower(City.name) == _normalise_text(city))

        stmt = stmt.order_by(Member.price.asc(), Shop.score.desc()).limit(limit)
        result = await session.execute(stmt)

    offers: list[MemberOffer] = []
    for random_key, price, shop_id, has_warranty, score, city_name in result:
        offers.append(
            MemberOffer(
                member_random_key=random_key,
                shop_id=int(shop_id),
                price=int(price),
                has_warranty=bool(has_warranty),
                shop_score=Decimal(score) if isinstance(score, (int, float, Decimal)) else None,
                city_name=city_name,
            )
        )

    return MemberFilterResponse(offers=offers)


CATEGORY_FEATURE_STATISTICS_TOOL = Tool(
    _category_feature_statistics,
    name="category_feature_statistics",
    description=(
        "Analyse a category or list of base products to understand which extra features "
        "appear most often. Provide either `category_hint` or `base_random_keys` and the tool "
        "returns up to twenty feature paths with representative values so you can craft "
        "targeted clarification questions."
    ),
)


FILTER_BASE_PRODUCTS_TOOL = Tool(
    _filter_base_products_by_constraints,
    name="filter_base_products_by_constraints",
    description=(
        "Filter catalogue base products using structured criteria such as category hints, "
        "brand preferences, price ranges, and feature requirements. The tool returns the "
        "strongest matches alongside evidence of which constraints they satisfied."
    ),
)


FILTER_MEMBERS_TOOL = Tool(
    _filter_members_by_constraints,
    name="filter_members_by_constraints",
    description=(
        "Given a resolved base product, retrieve shop-specific offers that honour price, "
        "warranty, score, or city requirements. Use this to pick the single best member "
        "once the correct base product is known."
    ),
)


__all__ = [
    "CATEGORY_FEATURE_STATISTICS_TOOL",
    "FILTER_BASE_PRODUCTS_TOOL",
    "FILTER_MEMBERS_TOOL",
]

