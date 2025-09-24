"""Data structures supporting the multi-turn member discovery flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

from pydantic import BaseModel, Field, conlist


class PriceRange(BaseModel):
    """Represents a desired price range for filtering members."""

    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)


class MemberFilters(BaseModel):
    """Aggregated constraints collected during the conversation."""

    text_queries: List[str] = Field(default_factory=list)
    category_id: int | None = None
    brand_id: int | None = None
    city_id: int | None = None
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    requires_warranty: bool | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=5.0)
    max_score: float | None = Field(default=None, ge=0.0, le=5.0)
    preferred_shop_ids: List[int] = Field(default_factory=list)
    allowed_shop_ids: List[int] = Field(default_factory=list)
    excluded_fields: set[str] = Field(default_factory=set)
    asked_questions: set[str] = Field(default_factory=set)
    last_question_key: str | None = None
    candidates_shown: List[str] = Field(default_factory=list)
    other_constraints: dict[str, Any] = Field(default_factory=dict)

    def add_text_queries(self, values: Iterable[str], *, limit: int = 8) -> None:
        """Append new fuzzy-search hints while capping the history length."""

        existing = list(self.text_queries)
        for value in values:
            trimmed = value.strip()
            if not trimmed:
                continue
            if trimmed in existing:
                continue
            existing.append(trimmed)
        if len(existing) > limit:
            existing = existing[-limit:]
        self.text_queries = existing


class ConstraintUpdate(BaseModel):
    """Output schema produced by the constraint extraction helper."""

    text_queries: List[str] = Field(default_factory=list)
    category_id: int | None = None
    brand_id: int | None = None
    city_id: int | None = None
    price_range: PriceRange | None = None
    requires_warranty: bool | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=5.0)
    max_score: float | None = Field(default=None, ge=0.0, le=5.0)
    preferred_shop_ids: List[int] = Field(default_factory=list)
    allowed_shop_ids: List[int] = Field(default_factory=list)
    selected_member_random_key: str | None = None
    excluded_fields: List[str] = Field(default_factory=list)
    cleared_fields: List[str] = Field(default_factory=list)
    rejected_candidates: bool = False
    notes: List[str] = Field(default_factory=list)


class MemberCandidate(BaseModel):
    """Single ranked member resulting from the discovery query."""

    member_random_key: str
    base_random_key: str
    base_name: str
    brand_id: int | None = None
    category_id: int | None = None
    shop_id: int
    price: int
    shop_score: float | None = None
    has_warranty: bool = False
    city_id: int | None = None
    city_name: str | None = None
    score: float = 0.0


class MemberSearchResult(BaseModel):
    """Lightweight container returned by the member search helper."""

    candidates: conlist(MemberCandidate, max_length=40) = Field(default_factory=list)


@dataclass(slots=True)
class MultiTurnState:
    """Conversation snapshot persisted across multi-turn requests."""

    chat_id: str
    filters: MemberFilters = field(default_factory=MemberFilters)
    turn_count: int = 0
    processed_message_count: int = 0
    pending_question_key: str | None = None
    presented_candidates: List[MemberCandidate] = field(default_factory=list)
    last_candidates: List[MemberCandidate] = field(default_factory=list)
    parser_history: Sequence[Any] = field(default_factory=tuple)
    completed: bool = False
    last_prompt: str | None = None

    def reset_presented_candidates(self) -> None:
        """Clear any previously surfaced candidate lists."""

        self.presented_candidates = []

