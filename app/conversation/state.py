"""In-memory state containers for routing and chat history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Literal

from cachetools import TTLCache

from app.agent.router import RouterRoute
from app.config import CACHE_TTL_SECONDS
from pydantic_ai.messages import ModelMessage


@dataclass(slots=True)
class ConversationMessage:
    """Represents a single utterance in a multi-turn conversation."""

    role: Literal["user", "assistant"]
    content: str
    base_random_keys: List[str] = field(default_factory=list)
    member_random_keys: List[str] = field(default_factory=list)


@dataclass(slots=True)
class _ConversationState:
    """Aggregates cached router outcomes and histories for a chat."""

    route: RouterRoute | None = None
    messages: List[ConversationMessage] = field(default_factory=list)
    model_messages: List[ModelMessage] = field(default_factory=list)


class ConversationStore:
    """Tracks router decisions and message history by chat identifier."""

    def __init__(self) -> None:
        # Allow a reasonable number of concurrent conversations while relying on
        # TTL eviction to keep the footprint bounded.
        self._state: TTLCache[str, _ConversationState] = TTLCache(
            maxsize=1024, ttl=CACHE_TTL_SECONDS
        )
        self._lock = asyncio.Lock()

    async def get_route(self, chat_id: str) -> RouterRoute | None:
        """Return the cached routing decision for a chat if available."""

        async with self._lock:
            state = self._state.get(chat_id)
            if state is None or state.route is None:
                return None
            # Refresh the TTL so active conversations stay warm.
            self._state[chat_id] = state
            return state.route

    async def remember_route(self, chat_id: str, route: RouterRoute) -> None:
        """Persist the routing decision for subsequent turns."""

        async with self._lock:
            state = self._state.get(chat_id) or _ConversationState()
            state.route = route
            self._state[chat_id] = state

    async def append_message(
        self, chat_id: str, message: ConversationMessage
    ) -> None:
        """Store a message in the chronological history for a chat."""

        async with self._lock:
            state = self._state.get(chat_id) or _ConversationState()
            state.messages.append(message)
            self._state[chat_id] = state

    async def get_history(self, chat_id: str) -> List[ConversationMessage]:
        """Return a copy of the stored message history for a chat."""

        async with self._lock:
            state = self._state.get(chat_id)
            if state is None or not state.messages:
                return []
            self._state[chat_id] = state
            return list(state.messages)

    async def get_model_history(self, chat_id: str) -> List[ModelMessage]:
        """Return the cached model-level history for a chat."""

        async with self._lock:
            state = self._state.get(chat_id)
            if state is None or not state.model_messages:
                return []
            self._state[chat_id] = state
            return list(state.model_messages)

    async def replace_model_history(
        self, chat_id: str, model_messages: List[ModelMessage]
    ) -> None:
        """Replace the stored model messages for a chat with a new snapshot."""

        async with self._lock:
            state = self._state.get(chat_id) or _ConversationState()
            state.model_messages = list(model_messages)
            self._state[chat_id] = state

    async def reset(self, chat_id: str) -> None:
        """Forget the cached routing choice and message history for a chat."""

        async with self._lock:
            self._state.pop(chat_id, None)

    async def reset_all(self) -> None:
        """Remove all cached routing decisions and histories."""

        async with self._lock:
            self._state.clear()


conversation_store = ConversationStore()


__all__ = ["ConversationMessage", "ConversationStore", "conversation_store"]
