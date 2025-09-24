"""State management primitives for the multi-turn member finder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .schemas import MemberCandidate


@dataclass
class MemberFilters:
    """Filters and hints that drive member retrieval."""

    text_queries: List[str] = field(default_factory=list)
    feature_hints: List[str] = field(default_factory=list)
    category_id: Optional[int] = None
    brand_id: Optional[int] = None
    preferred_shop_ids: List[int] = field(default_factory=list)
    allowed_shop_ids: List[int] = field(default_factory=list)
    city_id: Optional[int] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    requires_warranty: Optional[bool] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    def add_text_queries(self, values: List[str]) -> None:
        for value in values:
            trimmed = value.strip()
            if not trimmed:
                continue
            if trimmed not in self.text_queries:
                self.text_queries.append(trimmed)

    def add_feature_hints(self, values: List[str]) -> None:
        for value in values:
            trimmed = value.strip()
            if not trimmed:
                continue
            if trimmed not in self.feature_hints:
                self.feature_hints.append(trimmed)

    def summary(self) -> dict[str, object]:
        """Return a lightweight summary used for prompting."""

        return {
            "text_queries": self.text_queries,
            "feature_hints": self.feature_hints,
            "category_id": self.category_id,
            "brand_id": self.brand_id,
            "preferred_shop_ids": self.preferred_shop_ids,
            "allowed_shop_ids": self.allowed_shop_ids,
            "city_id": self.city_id,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "requires_warranty": self.requires_warranty,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass
class MemberSearchState:
    """Conversation-level state for member discovery."""

    filters: MemberFilters = field(default_factory=MemberFilters)
    excluded_fields: Set[str] = field(default_factory=set)
    asked_questions: Set[str] = field(default_factory=set)
    pending_question: Optional[str] = None
    candidates_shown: Set[str] = field(default_factory=set)
    turns_taken: int = 0
    last_candidates: List["MemberCandidate"] = field(default_factory=list)
    finalized_member_key: Optional[str] = None
    fallback_used: bool = False

    def to_prompt_payload(self) -> dict[str, object]:
        """Return a concise payload describing the current state."""

        return {
            "filters": self.filters.summary(),
            "excluded_fields": sorted(self.excluded_fields),
            "pending_question": self.pending_question,
            "asked_questions": sorted(self.asked_questions),
            "turns_taken": self.turns_taken,
        }


@dataclass
class MultiTurnSession:
    """Keeps per-chat caches for the multi-turn workflow."""

    chat_id: str
    state: MemberSearchState = field(default_factory=MemberSearchState)
    processed_message_count: int = 0
    constraint_history: List[ModelMessage] = field(default_factory=list)

    def reset(self) -> None:
        self.state = MemberSearchState()
        self.processed_message_count = 0
        self.constraint_history = []


