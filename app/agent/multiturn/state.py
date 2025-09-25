"""In-memory storage for multi-turn conversation state."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Dict

from .schemas import TurnState


class TurnStateStore:
    """Thread-safe in-memory map from chat identifiers to turn state."""

    def __init__(self) -> None:
        self._states: Dict[str, TurnState] = {}
        self._lock = asyncio.Lock()

    async def get(self, chat_id: str) -> TurnState | None:
        """Return a deep copy of the stored state for the chat, if any."""

        async with self._lock:
            state = self._states.get(chat_id)
            return state.model_copy(deep=True) if state is not None else None

    async def set(self, chat_id: str, state: TurnState) -> None:
        """Persist the provided state for subsequent turns."""

        async with self._lock:
            self._states[chat_id] = state.model_copy(deep=True)

    async def discard(self, chat_id: str) -> None:
        """Remove any stored state for the chat identifier."""

        async with self._lock:
            self._states.pop(chat_id, None)

    async def reset(self) -> None:
        """Clear all stored state (useful for testing)."""

        async with self._lock:
            self._states.clear()


@lru_cache(maxsize=1)
def get_turn_state_store() -> TurnStateStore:
    """Return the process-wide store used to persist turn state."""

    return TurnStateStore()


__all__ = ["TurnStateStore", "get_turn_state_store"]
