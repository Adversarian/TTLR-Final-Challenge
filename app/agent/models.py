"""Schema definitions for agent IO and tool responses."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Structured response returned by the LLM."""

    message: Optional[str] = Field(default=None)
    base_random_keys: Optional[List[str]] = Field(default=None, max_length=10)
    member_random_keys: Optional[List[str]] = Field(default=None, max_length=10)


class ProductResolveResult(BaseModel):
    """Output schema for product resolution tool."""

    base_random_key: Optional[str]
    candidates: List[dict] = Field(default_factory=list)


class FeatureLookupResult(BaseModel):
    """Output schema for attribute lookup tool."""

    value_text: Optional[str] = None
    raw_value: Optional[float] = None
    unit: Optional[str] = None
    provenance_key: Optional[str] = None
    needs_clarification: bool = False


class SellerStatsResult(BaseModel):
    """Output schema for seller stats tool."""

    result: Optional[str] = None
    member_random_key: Optional[str] = None
    shop_id: Optional[int] = None
