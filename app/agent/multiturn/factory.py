"""Factory for constructing the multi-turn dialogue coordinator."""

from __future__ import annotations

from functools import lru_cache

from .memory import ConversationMemory
from .orchestrator import MultiTurnCoordinator


@lru_cache(maxsize=1)
def get_multiturn_coordinator() -> MultiTurnCoordinator:
    """Return a singleton instance of the multi-turn coordinator."""

    memory = ConversationMemory()
    return MultiTurnCoordinator(memory=memory)


__all__ = ["get_multiturn_coordinator"]

