"""Schemas shared across the multi-turn workflow."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ConstraintUpdate(BaseModel):
    """Structured update produced by the constraint extractor."""

    text_queries: List[str] = Field(default_factory=list)
    feature_hints: List[str] = Field(default_factory=list)
    category_id: Optional[int] = None
    brand_id: Optional[int] = None
    preferred_shop_ids: List[int] = Field(default_factory=list)
    allowed_shop_ids: List[int] = Field(default_factory=list)
    city_id: Optional[int] = None
    min_price: Optional[int] = Field(default=None, ge=0)
    max_price: Optional[int] = Field(default=None, ge=0)
    requires_warranty: Optional[bool] = None
    min_score: Optional[float] = Field(default=None, ge=0, le=5)
    max_score: Optional[float] = Field(default=None, ge=0, le=5)
    excluded_fields: List[str] = Field(default_factory=list)
    clear_fields: List[str] = Field(default_factory=list)
    selected_member_random_key: Optional[str] = None
    notes: Optional[str] = None


class MemberCandidate(BaseModel):
    """Represents one shop/base-product combination."""

    member_random_key: str
    base_random_key: str
    base_name: str
    shop_id: int
    city_id: Optional[int]
    city_name: Optional[str]
    price: int
    shop_score: float
    has_warranty: bool
    brand_id: Optional[int]
    category_id: Optional[int]
    text_score: float
    score: float


class MemberSearchResult(BaseModel):
    """Payload returned by the member search tool."""

    candidates: List[MemberCandidate]
