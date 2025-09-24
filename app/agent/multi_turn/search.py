"""Database access helpers for the multi-turn member discovery flow."""

from __future__ import annotations

from typing import Iterable, List

from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import BaseProduct, City, Member, Shop
from .models import MemberCandidate, MemberFilters, MemberSearchResult


def _normalise_text(value: str) -> str:
    """Mirror the lightweight normalisation logic used in product search."""

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


def _prepare_query_text(text_queries: Iterable[str]) -> str:
    """Collapse collected search hints into a single fuzzy string."""

    seen = []
    for value in text_queries:
        trimmed = value.strip()
        if not trimmed:
            continue
        if trimmed in seen:
            continue
        seen.append(trimmed)
    return " ".join(seen)


async def search_members(
    session: AsyncSession,
    filters: MemberFilters,
    *,
    limit: int = 20,
) -> MemberSearchResult:
    """Return ranked member candidates that satisfy the collected filters."""

    query_text = _prepare_query_text(filters.text_queries)
    normalized_query = _normalise_text(query_text) if query_text else ""

    extra_features_text = func.cast(BaseProduct.extra_features, BaseProduct.persian_name.type)
    name_similarity = func.greatest(
        func.similarity(BaseProduct.persian_name, normalized_query),
        func.similarity(func.coalesce(BaseProduct.english_name, ""), normalized_query),
            func.similarity(extra_features_text, normalized_query),
    )
    text_score = name_similarity if normalized_query else literal(0.0)

    score_component = func.coalesce(Shop.score, 0.0) / 5.0

    target_price = None
    if filters.min_price is not None and filters.max_price is not None:
        target_price = (filters.min_price + filters.max_price) / 2
    elif filters.min_price is not None:
        target_price = filters.min_price
    elif filters.max_price is not None:
        target_price = filters.max_price

    if target_price and target_price > 0:
        price_distance = func.abs(Member.price - target_price)
        price_component = func.greatest(0.0, 1.0 - price_distance / target_price)
    else:
        price_component = literal(0.0)

    warranty_bonus = literal(0.0)
    if filters.requires_warranty is True:
        warranty_bonus = case((Shop.has_warranty.is_(True), literal(0.15)), else_=literal(0.0))
    elif filters.requires_warranty is None:
        warranty_bonus = case((Shop.has_warranty.is_(True), literal(0.05)), else_=literal(0.0))

    preferred_bonus = literal(0.0)
    if filters.preferred_shop_ids:
        preferred_bonus = case(
            (Member.shop_id.in_(filters.preferred_shop_ids), literal(0.1)),
            else_=literal(0.0),
        )

    composite_score = (
        text_score * 0.55
        + price_component * 0.2
        + score_component * 0.2
        + warranty_bonus
        + preferred_bonus
    ).label("rank_score")

    stmt = (
        select(
            Member.random_key,
            Member.base_random_key,
            BaseProduct.persian_name,
            BaseProduct.brand_id,
            BaseProduct.category_id,
            Member.shop_id,
            Member.price,
            Shop.score,
            Shop.has_warranty,
            Shop.city_id,
            City.name,
            composite_score,
        )
        .select_from(Member)
        .join(BaseProduct, Member.base_random_key == BaseProduct.random_key)
        .join(Shop, Member.shop_id == Shop.id)
        .outerjoin(City, Shop.city_id == City.id)
    )

    conditions = []
    if filters.category_id is not None:
        conditions.append(BaseProduct.category_id == filters.category_id)
    if filters.brand_id is not None:
        conditions.append(BaseProduct.brand_id == filters.brand_id)
    if filters.city_id is not None:
        conditions.append(Shop.city_id == filters.city_id)
    if filters.min_price is not None:
        conditions.append(Member.price >= filters.min_price)
    if filters.max_price is not None:
        conditions.append(Member.price <= filters.max_price)
    if filters.requires_warranty is True:
        conditions.append(Shop.has_warranty.is_(True))
    if filters.requires_warranty is False:
        conditions.append(Shop.has_warranty.is_(False))
    if filters.min_score is not None:
        conditions.append(Shop.score >= filters.min_score)
    if filters.max_score is not None:
        conditions.append(Shop.score <= filters.max_score)
    if filters.allowed_shop_ids:
        conditions.append(Member.shop_id.in_(filters.allowed_shop_ids))
    if filters.preferred_shop_ids:
        conditions.append(Member.shop_id.in_(filters.preferred_shop_ids))

    if normalized_query:
        trigram_predicate = or_(
            BaseProduct.persian_name.op("%")(normalized_query),
            func.coalesce(BaseProduct.english_name, "").op("%")(normalized_query),
            extra_features_text.op("%")(normalized_query),
        )
        conditions.append(trigram_predicate)

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(composite_score.desc(), Member.price.asc()).limit(limit)

    result = await session.execute(stmt)

    candidates: List[MemberCandidate] = []
    for row in result.all():
        (
            member_key,
            base_key,
            base_name,
            brand_id,
            category_id,
            shop_id,
            price,
            shop_score,
            has_warranty,
            city_id,
            city_name,
            rank_score,
        ) = row
        candidates.append(
            MemberCandidate(
                member_random_key=member_key,
                base_random_key=base_key,
                base_name=base_name,
                brand_id=brand_id,
                category_id=category_id,
                shop_id=shop_id,
                price=price,
                shop_score=float(shop_score) if shop_score is not None else None,
                has_warranty=bool(has_warranty),
                city_id=city_id,
                city_name=city_name,
                score=float(rank_score or 0.0),
            )
        )

    return MemberSearchResult(candidates=candidates)


__all__ = ["search_members"]

