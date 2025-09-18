"""Shared data structures for the Torob shopping assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from llama_index.core.workflow import StartEvent, StopEvent
from pydantic import BaseModel, Field

from app.models.chat import ChatRequest, ChatResponse


@dataclass
class ProductCandidate:
    """Lightweight representation of a product candidate prior to hydration."""

    random_key: str
    persian_name: Optional[str] = None
    english_name: Optional[str] = None
    matched_via: str = "semantic"
    score: Optional[float] = None


class ProductLookupArgs(BaseModel):
    """Arguments accepted by the ``lookup_products`` function tool."""

    product_name: Optional[str] = Field(
        default=None, description="Free-form text describing the product."
    )
    base_random_key: Optional[str] = Field(
        default=None, description="Known Torob base random key."
    )
    member_random_key: Optional[str] = Field(
        default=None, description="Known Torob member random key."
    )
    limit: int = Field(default=5, ge=1, le=20)


class SellerStats(BaseModel):
    offer_count: int = 0
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    avg_price: Optional[float] = None


class SellerOffer(BaseModel):
    member_random_key: Optional[str] = None
    price: Optional[float] = None
    shop_id: Optional[int] = None
    shop_score: Optional[float] = None
    has_warranty: Optional[bool] = None
    city_id: Optional[int] = None
    city_name: Optional[str] = None


class ProductContext(BaseModel):
    random_key: str
    persian_name: Optional[str] = None
    english_name: Optional[str] = None
    category_id: Optional[int] = None
    category_title: Optional[str] = None
    brand_id: Optional[int] = None
    brand_title: Optional[str] = None
    matched_via: str
    match_score: Optional[float] = None
    features: Dict[str, str] = Field(default_factory=dict)
    feature_list: List[str] = Field(default_factory=list)
    seller_stats: SellerStats = Field(default_factory=SellerStats)
    top_offers: List[SellerOffer] = Field(default_factory=list)


class StructuredChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None


class ChatWorkflowInput(StartEvent):
    request: ChatRequest


class ChatWorkflowOutput(StopEvent):
    response: ChatResponse


__all__ = [
    "ChatWorkflowInput",
    "ChatWorkflowOutput",
    "ProductCandidate",
    "ProductContext",
    "ProductLookupArgs",
    "SellerOffer",
    "SellerStats",
    "StructuredChatResponse",
]
