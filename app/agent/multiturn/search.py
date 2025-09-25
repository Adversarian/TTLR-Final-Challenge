"""Candidate search and lightweight ranking for the multi-turn flow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Sequence

from pydantic import BaseModel, Field
from sqlalchemy import (
    Float,
    Integer,
    Select,
    Text,
    cast,
    func,
    literal,
    select,
)
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import BaseProduct, Brand, Category, City, Member, Shop
from .contracts import MemberDetails


_TOP_CANDIDATES = 5
_DISTRIBUTION_LIMIT = 6
_PRICE_BUCKET_SIZE = 500_000


class DistributionValue(BaseModel):
    """Represents the frequency of a specific attribute value."""

    value: str = Field(..., description="Display label for the attribute bucket.")
    count: int = Field(..., description="Number of matching candidates in the bucket.")


class AttributeDistributions(BaseModel):
    """Summary of attribute frequencies across the filtered candidate set."""

    brand: list[DistributionValue] = Field(default_factory=list)
    city: list[DistributionValue] = Field(default_factory=list)
    price_band: list[DistributionValue] = Field(default_factory=list)
    warranty: list[DistributionValue] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    """Top-ranked offer returned by the candidate search."""

    member_random_key: str
    base_random_key: str
    product_name: str
    brand_name: str | None = None
    city_name: str | None = None
    price: int | None = None
    shop_score: float | None = None
    has_warranty: bool | None = None
    relevance: float = 0.0
    label: str = ""


class CandidateSearchResult(BaseModel):
    """Compact payload returned by the candidate search routine."""

    count: int = 0
    candidates: list[RankedCandidate] = Field(default_factory=list)
    distributions: AttributeDistributions = Field(default_factory=AttributeDistributions)


@dataclass(slots=True)
class _SearchStatement:
    """Container bundling the compiled SQL statement and limits used."""

    statement: Select
    top_limit: int
    distribution_limit: int


def _normalize_names(values: Iterable[str]) -> set[str]:
    """Lower-case and strip names to align with catalogue storage."""

    normalised: set[str] = set()
    for value in values:
        if not value:
            continue
        trimmed = value.strip().lower()
        if trimmed:
            normalised.add(trimmed)
    return normalised


def _compose_keyword_query(details: MemberDetails) -> str:
    """Merge keyword and attribute hints into a single search phrase."""

    tokens: list[str] = []
    for token in details.keywords:
        trimmed = token.strip()
        if trimmed:
            tokens.append(trimmed)
    for value in details.product_attributes.values():
        trimmed = value.strip()
        if trimmed:
            tokens.append(trimmed)
    return " ".join(tokens)


def _build_candidate_search_statement(
    details: MemberDetails,
    *,
    top_limit: int = _TOP_CANDIDATES,
    distribution_limit: int = _DISTRIBUTION_LIMIT,
) -> _SearchStatement:
    """Construct the SQL statement that powers the candidate search."""

    excluded = details.excluded_fields

    filters: list = []

    brand_filters = _normalize_names(details.brand_names)
    if brand_filters and "brand" not in excluded:
        filters.append(func.lower(Brand.title).in_(brand_filters))

    category_filters = _normalize_names(details.category_names)
    needs_category_join = bool(category_filters and "category" not in excluded)
    if category_filters and "category" not in excluded:
        filters.append(func.lower(Category.title).in_(category_filters))

    city_filters = _normalize_names(details.city_names)
    if city_filters and "city" not in excluded:
        filters.append(func.lower(City.name).in_(city_filters))

    if details.min_price is not None and "price" not in excluded:
        filters.append(Member.price >= details.min_price)
    if details.max_price is not None and "price" not in excluded:
        filters.append(Member.price <= details.max_price)

    if details.min_shop_score is not None and "score" not in excluded:
        filters.append(cast(Shop.score, Float) >= details.min_shop_score)

    if details.warranty_required is not None and "warranty" not in excluded:
        filters.append(Shop.has_warranty.is_(bool(details.warranty_required)))

    keyword_query = _compose_keyword_query(details)

    if keyword_query:
        ts_query = func.websearch_to_tsquery("simple", keyword_query)
        fts_rank = func.coalesce(func.ts_rank_cd(BaseProduct.search_vector, ts_query), 0.0)
        name_similarity = func.coalesce(func.similarity(BaseProduct.persian_name, keyword_query), 0.0)
        feature_similarity = func.coalesce(
            func.similarity(cast(BaseProduct.extra_features, Text), keyword_query),
            0.0,
        )
    else:
        fts_rank = literal(0.0)
        name_similarity = literal(0.0)
        feature_similarity = literal(0.0)

    score_expr = (
        (fts_rank * 0.6)
        + (name_similarity * 0.25)
        + (feature_similarity * 0.1)
        + (func.coalesce(cast(Shop.score, Float), 0.0) * 0.05)
    )

    price_bucket = cast(
        func.floor(Member.price / literal(_PRICE_BUCKET_SIZE)),
        Integer,
    ).label("price_bucket")

    base_query = (
        select(
            Member.random_key.label("member_random_key"),
            BaseProduct.random_key.label("base_random_key"),
            BaseProduct.persian_name.label("product_name"),
            Brand.title.label("brand_name"),
            City.name.label("city_name"),
            Member.price.label("price"),
            Shop.score.label("shop_score"),
            Shop.has_warranty.label("has_warranty"),
            score_expr.label("relevance"),
            func.count().over().label("total_count"),
            func.row_number().over(order_by=score_expr.desc()).label("row_number"),
            price_bucket,
        )
        .select_from(Member)
        .join(BaseProduct, Member.base_random_key == BaseProduct.random_key)
        .join(Shop, Member.shop_id == Shop.id)
        .join(City, Shop.city_id == City.id)
        .outerjoin(Brand, BaseProduct.brand_id == Brand.id)
    )

    if needs_category_join:
        base_query = base_query.join(Category, BaseProduct.category_id == Category.id)

    if filters:
        base_query = base_query.where(*filters)

    filtered = base_query.cte("filtered")

    total_count_subq = select(func.count()).select_from(filtered).scalar_subquery()

    candidate_rows_subq = (
        select(
            func.array_agg(
                aggregate_order_by(
                    func.row(
                        filtered.c.member_random_key,
                        filtered.c.base_random_key,
                        filtered.c.product_name,
                        filtered.c.brand_name,
                        filtered.c.city_name,
                        filtered.c.price,
                        filtered.c.shop_score,
                        filtered.c.has_warranty,
                        filtered.c.relevance,
                    ),
                    filtered.c.row_number,
                )
            )
        )
        .select_from(filtered)
        .where(filtered.c.row_number <= top_limit)
        .scalar_subquery()
    )

    brand_value = func.coalesce(filtered.c.brand_name, literal("نامشخص"))
    brand_counts = (
        select(
            brand_value.label("value"),
            func.count().label("count"),
        )
        .group_by(brand_value)
        .order_by(func.count().desc(), brand_value)
        .limit(distribution_limit)
        .cte("brand_counts")
    )
    brand_array_subq = (
        select(
            func.array_agg(
                aggregate_order_by(
                    func.row(brand_counts.c.value, brand_counts.c.count),
                    brand_counts.c.count.desc(),
                )
            )
        )
        .select_from(brand_counts)
        .scalar_subquery()
    )

    city_value = func.coalesce(filtered.c.city_name, literal("نامشخص"))
    city_counts = (
        select(
            city_value.label("value"),
            func.count().label("count"),
        )
        .group_by(city_value)
        .order_by(func.count().desc(), city_value)
        .limit(distribution_limit)
        .cte("city_counts")
    )
    city_array_subq = (
        select(
            func.array_agg(
                aggregate_order_by(
                    func.row(city_counts.c.value, city_counts.c.count),
                    city_counts.c.count.desc(),
                )
            )
        )
        .select_from(city_counts)
        .scalar_subquery()
    )

    price_counts = (
        select(
            filtered.c.price_bucket.label("bucket"),
            func.count().label("count"),
            func.min(filtered.c.price).label("min_price"),
            func.max(filtered.c.price).label("max_price"),
        )
        .group_by(filtered.c.price_bucket)
        .order_by(func.count().desc(), filtered.c.price_bucket)
        .limit(distribution_limit)
        .cte("price_counts")
    )
    price_array_subq = (
        select(
            func.array_agg(
                aggregate_order_by(
                    func.row(
                        price_counts.c.bucket,
                        price_counts.c.count,
                        price_counts.c.min_price,
                        price_counts.c.max_price,
                    ),
                    price_counts.c.count.desc(),
                )
            )
        )
        .select_from(price_counts)
        .scalar_subquery()
    )

    warranty_counts = (
        select(
            filtered.c.has_warranty.label("value"),
            func.count().label("count"),
        )
        .group_by(filtered.c.has_warranty)
        .order_by(func.count().desc(), filtered.c.has_warranty)
        .limit(2)
        .cte("warranty_counts")
    )
    warranty_array_subq = (
        select(
            func.array_agg(
                aggregate_order_by(
                    func.row(warranty_counts.c.value, warranty_counts.c.count),
                    warranty_counts.c.count.desc(),
                )
            )
        )
        .select_from(warranty_counts)
        .scalar_subquery()
    )

    statement = select(
        func.coalesce(total_count_subq, 0).label("total_count"),
        candidate_rows_subq.label("candidates"),
        brand_array_subq.label("brand_counts"),
        city_array_subq.label("city_counts"),
        price_array_subq.label("price_counts"),
        warranty_array_subq.label("warranty_counts"),
    )

    return _SearchStatement(statement=statement, top_limit=top_limit, distribution_limit=distribution_limit)


def _format_price(value: int | None) -> str:
    """Return a human-friendly price string for labels."""

    if value is None:
        return "نامشخص"
    return f"{int(value):,}"


def _format_score(score: float | None) -> str:
    """Format the shop score for display in option labels."""

    if score is None:
        return "—"
    return f"{score:.1f}"


def _compose_label(candidate: RankedCandidate) -> str:
    """Assemble the user-facing label for a ranked candidate."""

    brand = candidate.brand_name or "بدون برند"
    price_text = _format_price(candidate.price)
    score_text = _format_score(candidate.shop_score)
    return (
        f"«{candidate.product_name} — {brand} — {price_text} تومان — "
        f"فروشنده امتیاز {score_text}»"
    )


def _coerce_int(value: object | None) -> int | None:
    """Convert numeric database values into Python integers."""

    if value is None:
        return None
    if isinstance(value, Decimal):
        return int(value)
    return int(value)


def _coerce_float(value: object | None) -> float | None:
    """Convert numeric database values into floats."""

    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _deserialize_candidate_rows(data: Sequence[Sequence[object]] | None) -> list[RankedCandidate]:
    """Transform the raw SQL payload into structured candidates."""

    if not data:
        return []

    candidates: list[RankedCandidate] = []
    for entry in data:
        (
            member_rk,
            base_rk,
            product_name,
            brand_name,
            city_name,
            price,
            shop_score,
            has_warranty,
            relevance,
        ) = entry

        price_value = _coerce_int(price)
        score_value = _coerce_float(shop_score)
        relevance_value = _coerce_float(relevance) or 0.0

        candidate = RankedCandidate(
            member_random_key=member_rk,
            base_random_key=base_rk,
            product_name=product_name,
            brand_name=brand_name,
            city_name=city_name,
            price=price_value,
            shop_score=score_value,
            has_warranty=bool(has_warranty) if has_warranty is not None else None,
            relevance=relevance_value,
        )
        candidate.label = _compose_label(candidate)
        candidates.append(candidate)

    return candidates


def _deserialize_distribution(
    rows: Sequence[Sequence[object]] | None,
    *,
    formatter: Callable[[Sequence[object]], str],
) -> list[DistributionValue]:
    """Convert aggregated counts into :class:`DistributionValue` rows."""

    if not rows:
        return []

    values: list[DistributionValue] = []
    for row in rows:
        bucket_value = formatter(row)
        count = int(row[1])
        values.append(DistributionValue(value=bucket_value, count=count))
    return values


def _format_price_band(row: Sequence[object]) -> str:
    """Return a Persian-friendly price band label."""

    min_price = _coerce_int(row[2])
    max_price = _coerce_int(row[3])
    if min_price is None and max_price is None:
        return "نامشخص"
    if min_price is None:
        return f"تا {max_price:,}"
    if max_price is None or max_price == min_price:
        return f"حدود {min_price:,}"
    return f"{min_price:,} تا {max_price:,}"


def _format_boolean_bucket(row: Sequence[object]) -> str:
    """Map boolean warranty buckets to Persian-friendly labels."""

    value = row[0]
    if value is True:
        return "گارانتی دارد"
    if value is False:
        return "بدون گارانتی"
    return "نامشخص"


def _format_plain_bucket(row: Sequence[object]) -> str:
    """Return the first column as a trimmed string label."""

    value = row[0]
    if value is None:
        return "نامشخص"
    text_value = str(value).strip()
    return text_value or "نامشخص"


async def search_candidates(
    session: AsyncSession,
    details: MemberDetails,
    *,
    top_limit: int = _TOP_CANDIDATES,
    distribution_limit: int = _DISTRIBUTION_LIMIT,
) -> CandidateSearchResult:
    """Execute the candidate search against PostgreSQL."""

    compiled = _build_candidate_search_statement(
        details,
        top_limit=top_limit,
        distribution_limit=distribution_limit,
    )

    result = await session.execute(compiled.statement)
    row = result.one()

    candidates = _deserialize_candidate_rows(row.candidates)

    distributions = AttributeDistributions(
        brand=_deserialize_distribution(row.brand_counts, formatter=_format_plain_bucket),
        city=_deserialize_distribution(row.city_counts, formatter=_format_plain_bucket),
        price_band=_deserialize_distribution(row.price_counts, formatter=_format_price_band),
        warranty=_deserialize_distribution(row.warranty_counts, formatter=_format_boolean_bucket),
    )

    return CandidateSearchResult(
        count=int(row.total_count or 0),
        candidates=candidates,
        distributions=distributions,
    )


__all__ = [
    "AttributeDistributions",
    "CandidateSearchResult",
    "DistributionValue",
    "RankedCandidate",
    "search_candidates",
]
