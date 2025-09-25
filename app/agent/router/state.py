"""In-memory storage for cached router decisions."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Dict, Optional


class RouterDecisionStore:
    """Thread-safe mapping from chat identifiers to routing decisions."""

    def __init__(self) -> None:
        self._routes: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def get(self, chat_id: str) -> Optional[str]:
        """Return the cached route for the chat identifier, if any."""

        async with self._lock:
            return self._routes.get(chat_id)

    async def set(self, chat_id: str, route: str) -> None:
        """Persist the router decision for the chat identifier."""

        async with self._lock:
            self._routes[chat_id] = route

    async def discard(self, chat_id: str) -> None:
        """Remove any cached route for the chat identifier."""

        async with self._lock:
            self._routes.pop(chat_id, None)

    async def reset(self) -> None:
        """Clear all cached routes (useful for tests)."""

        async with self._lock:
            self._routes.clear()


@lru_cache(maxsize=1)
def get_router_decision_store() -> RouterDecisionStore:
    """Return the process-wide router decision store."""

    return RouterDecisionStore()


__all__ = ["RouterDecisionStore", "get_router_decision_store"]
