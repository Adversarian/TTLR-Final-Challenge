"""Structured models used by the multi-turn member selection agent."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class SearchCandidate(BaseModel):
    """Represents a single shop offering returned by the search tool."""

    member_random_key: str
    base_name: str
    brand: Optional[str] = None
    price: int
    shop_name: str
    shop_score: Optional[float] = None
    relevance: Optional[float] = None


class SearchMembersDistributions(BaseModel):
    """Frequency breakdowns that help the agent pick the next question."""

    brand: Optional[List[Tuple[Optional[int], int]]] = None
    city: Optional[List[Tuple[Optional[int], int]]] = None
    price_band: Optional[List[Tuple[str, int]]] = None
    warranty: Optional[List[Tuple[bool, int]]] = None


class SearchMembersResult(BaseModel):
    """Payload returned from the member search database tool."""

    count: int = Field(..., ge=0)
    topK: List[SearchCandidate] = Field(default_factory=list, max_length=10)
    distributions: SearchMembersDistributions = Field(
        default_factory=SearchMembersDistributions
    )


class TurnFilters(BaseModel):
    """Hard constraints applied to the member search."""

    brand_id: Optional[int] = None
    brand_name: Optional[str] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    city_id: Optional[int] = None
    city_name: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    has_warranty: Optional[bool] = None
    shop_min_score: Optional[float] = None


class CandidateOption(BaseModel):
    """Compact option presented to the user for numeric selection."""

    idx: int = Field(..., ge=1, le=10)
    label: str
    member_random_key: str


class TurnState(BaseModel):
    """Minimal state persisted between multi-turn interactions."""

    turn: int = 1
    filters: TurnFilters = Field(default_factory=TurnFilters)
    query_tokens: List[str] = Field(default_factory=list)
    asked_fields: List[str] = Field(default_factory=list)
    excluded_fields: List[str] = Field(default_factory=list)
    last_options: List[CandidateOption] = Field(default_factory=list, max_length=10)
    pending_question: Optional[str] = None


class MultiTurnAgentInput(BaseModel):
    """Input payload provided to the multi-turn agent each turn."""

    chat_id: str
    state: TurnState
    user_message: str
    normalized_message: str


class MultiTurnAgentReply(BaseModel):
    """Structured output expected from the multi-turn agent."""

    message: Optional[str] = None
    member_random_key: Optional[str] = None
    done: bool = False
    updated_state: TurnState
    action: Literal["ask", "return", "clarify"]


__all__ = [
    "CandidateOption",
    "MultiTurnAgentInput",
    "MultiTurnAgentReply",
    "SearchCandidate",
    "SearchMembersDistributions",
    "SearchMembersResult",
    "TurnFilters",
    "TurnState",
]
