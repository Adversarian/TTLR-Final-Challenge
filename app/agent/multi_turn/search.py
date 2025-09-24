"""Database helpers for member retrieval."""

from __future__ import annotations

from typing import Iterable, List

from sqlalchemy import Text, cast, case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...models import BaseProduct, City, Member, Shop
from .schemas import MemberCandidate, MemberSearchResult
from .state import MemberFilters


def _normalise_terms(terms: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for term in terms:
        trimmed = term.strip().lower()
        if trimmed:
            cleaned.append(trimmed)
    return cleaned


async def search_members(
    session: AsyncSession,
    filters: MemberFilters,
    limit: int = 10,
) -> MemberSearchResult:
    """Search members with soft textual matching and weighted scoring."""

    terms = _normalise_terms(filters.text_queries + filters.feature_hints)

    similarity_exprs = []
    threshold = settings.search_similarity_threshold * 0.6

    features_text = cast(BaseProduct.extra_features, Text)
    for term in terms:
        similarity_exprs.append(
            func.greatest(
                func.similarity(BaseProduct.persian_name, term),
                func.similarity(func.coalesce(BaseProduct.english_name, ""), term),
                func.similarity(func.coalesce(features_text, ""), term),
            )
        )

    if similarity_exprs:
        total_expr = similarity_exprs[0]
        for expr in similarity_exprs[1:]:
            total_expr = total_expr + expr
        text_score_expr = total_expr / len(similarity_exprs)
        text_predicate = or_(*(expr >= threshold for expr in similarity_exprs))
    else:
        text_score_expr = literal(0.0)
        text_predicate = None

    warranty_bonus = case((Shop.has_warranty.is_(True), literal(0.1)), else_=literal(0.0))
    score_component = func.coalesce(Shop.score / 5.0, 0.0) * 0.2
    base_score_expr = text_score_expr * 0.7 + score_component + warranty_bonus

    base_query = (
        select(
            Member.random_key.label("member_random_key"),
            Member.price,
            Shop.id.label("shop_id"),
            Shop.score,
            Shop.has_warranty,
            Shop.city_id,
            BaseProduct.random_key.label("base_random_key"),
            BaseProduct.persian_name.label("base_name"),
            BaseProduct.brand_id,
            BaseProduct.category_id,
            City.name.label("city_name"),
            text_score_expr.label("text_score"),
            base_score_expr.label("score"),
        )
        .join(Shop, Member.shop_id == Shop.id)
        .join(BaseProduct, Member.base_random_key == BaseProduct.random_key)
        .outerjoin(City, Shop.city_id == City.id)
    )

    if text_predicate is not None:
        base_query = base_query.where(text_predicate)

    if filters.category_id is not None:
        base_query = base_query.where(BaseProduct.category_id == filters.category_id)

    if filters.brand_id is not None:
        base_query = base_query.where(BaseProduct.brand_id == filters.brand_id)

    if filters.city_id is not None:
        base_query = base_query.where(Shop.city_id == filters.city_id)

    if filters.preferred_shop_ids:
        base_query = base_query.where(Shop.id.in_(filters.preferred_shop_ids))

    if filters.allowed_shop_ids:
        base_query = base_query.where(Shop.id.in_(filters.allowed_shop_ids))

    if filters.min_price is not None:
        base_query = base_query.where(Member.price >= filters.min_price)

    if filters.max_price is not None:
        base_query = base_query.where(Member.price <= filters.max_price)

    if filters.requires_warranty is True:
        base_query = base_query.where(Shop.has_warranty.is_(True))

    if filters.min_score is not None:
        base_query = base_query.where(Shop.score >= filters.min_score)

    if filters.max_score is not None:
        base_query = base_query.where(Shop.score <= filters.max_score)

    scored = base_query.cte("scored_members")

    if terms:
        ordering = scored.c.score.desc()
    else:
        ordering = scored.c.price.asc()

    stmt = select(scored).order_by(ordering).limit(limit)

    result = await session.execute(stmt)

    candidates = [
        MemberCandidate(
            member_random_key=row.member_random_key,
            base_random_key=row.base_random_key,
            base_name=row.base_name,
            shop_id=row.shop_id,
            city_id=row.city_id,
            city_name=row.city_name,
            price=int(row.price),
            shop_score=float(row.score or 0.0),
            has_warranty=bool(row.has_warranty),
            brand_id=row.brand_id,
            category_id=row.category_id,
            text_score=float(row.text_score or 0.0),
            score=float(row.score or 0.0),
        )
        for row in result
    ]

    candidates.sort(key=lambda item: item.score, reverse=True)
    return MemberSearchResult(candidates=candidates)
