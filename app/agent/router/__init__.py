"""Lightweight routing agent for selecting the chat strategy."""

from __future__ import annotations

from .factory import get_router_agent
from .schemas import RoutingDecision

__all__ = [
    "RoutingDecision",
    "get_router_agent",
]
