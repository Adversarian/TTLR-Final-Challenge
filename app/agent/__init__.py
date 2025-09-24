"""Agent package exposing the public interface for the shopping assistant."""

from __future__ import annotations

from .dependencies import AgentDependencies
from .factory import get_agent
from .multi_turn import MultiTurnManager, get_multi_turn_manager
from .router import RouterDecision, get_router
from .schemas import (
    AgentReply,
    CitySellerStatistics,
    FeatureLookupResult,
    ProductFeature,
    ProductMatch,
    ProductSearchResult,
    SellerStatistics,
)
from .tools import (
    FEATURE_LOOKUP_TOOL,
    PRODUCT_SEARCH_TOOL,
    SELLER_STATISTICS_TOOL,
    _fetch_feature_details,
)

__all__ = [
    "AgentDependencies",
    "AgentReply",
    "CitySellerStatistics",
    "FeatureLookupResult",
    "ProductFeature",
    "ProductMatch",
    "ProductSearchResult",
    "SellerStatistics",
    "RouterDecision",
    "MultiTurnManager",
    "FEATURE_LOOKUP_TOOL",
    "PRODUCT_SEARCH_TOOL",
    "SELLER_STATISTICS_TOOL",
    "_fetch_feature_details",
    "get_agent",
    "get_multi_turn_manager",
    "get_router",
]
