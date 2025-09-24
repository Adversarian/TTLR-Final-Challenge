"""Agent package exposing the public interface for the shopping assistant."""

from __future__ import annotations

from .dependencies import AgentDependencies
from .factory import get_agent
from .multi_turn.coordinator import (
    Scenario4Coordinator,
    get_scenario4_coordinator,
)
from .router import RouterDecision, RouterReply, get_router
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
    "FEATURE_LOOKUP_TOOL",
    "PRODUCT_SEARCH_TOOL",
    "SELLER_STATISTICS_TOOL",
    "_fetch_feature_details",
    "get_agent",
    "get_router",
    "RouterDecision",
    "RouterReply",
    "Scenario4Coordinator",
    "get_scenario4_coordinator",
]
