"""Agent package exposing the public interface for the shopping assistant."""

from __future__ import annotations

from .dependencies import AgentDependencies
from .factory import get_agent
from .multi_turn import get_multi_turn_agent
from .schemas import (
    AgentReply,
    CitySellerStatistics,
    FeatureLookupResult,
    ProductFeature,
    ProductMatch,
    ProductSearchResult,
    SellerCandidateSummary,
    SellerCandidateSummaryList,
    SellerStatistics,
)
from .tools import (
    FEATURE_LOOKUP_TOOL,
    FEATURE_LIST_FOR_BASES_TOOL,
    PRODUCT_SEARCH_TOOL,
    PRODUCT_SEARCH_WITH_FEATURES_TOOL,
    SELLER_CANDIDATE_SUMMARY_TOOL,
    SELLER_OFFERS_TOOL,
    SELLER_STATISTICS_TOOL,
    _fetch_feature_details,
)
from .router import RouterDecision, RouterRoute, get_router_agent

__all__ = [
    "AgentDependencies",
    "AgentReply",
    "CitySellerStatistics",
    "FeatureLookupResult",
    "FEATURE_LIST_FOR_BASES_TOOL",
    "ProductFeature",
    "ProductMatch",
    "ProductSearchResult",
    "PRODUCT_SEARCH_WITH_FEATURES_TOOL",
    "SellerCandidateSummary",
    "SellerCandidateSummaryList",
    "SellerStatistics",
    "FEATURE_LOOKUP_TOOL",
    "PRODUCT_SEARCH_TOOL",
    "SELLER_CANDIDATE_SUMMARY_TOOL",
    "SELLER_OFFERS_TOOL",
    "SELLER_STATISTICS_TOOL",
    "_fetch_feature_details",
    "get_agent",
    "get_multi_turn_agent",
    "get_router_agent",
    "RouterDecision",
    "RouterRoute",
]
