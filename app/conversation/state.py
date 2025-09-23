"""In-memory state containers for routing and chat history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from typing import Dict, List, Literal

from app.agent.router import RouterRoute
from pydantic_ai.messages import ModelMessage


@dataclass(slots=True)
class ConversationMessage:
    """Represents a single utterance in a multi-turn conversation."""

    role: Literal["user", "assistant"]
    content: str
    base_random_keys: List[str] = field(default_factory=list)
    member_random_keys: List[str] = field(default_factory=list)


_CACHE_TTL_SECONDS = 120.0


class ConversationStore:
    """Tracks router decisions and message history by chat identifier."""

    def __init__(self) -> None:
        self._routes: Dict[str, RouterRoute] = {}
        self._messages: Dict[str, List[ConversationMessage]] = {}
        self._model_messages: Dict[str, List[ModelMessage]] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get_route(self, chat_id: str) -> RouterRoute | None:
        """Return the cached routing decision for a chat if available."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            route = self._routes.get(chat_id)
            if route is not None:
                self._last_access[chat_id] = now
            return route

    async def remember_route(self, chat_id: str, route: RouterRoute) -> None:
        """Persist the routing decision for subsequent turns."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            self._routes[chat_id] = route
            self._last_access[chat_id] = now

    async def append_message(
        self, chat_id: str, message: ConversationMessage
    ) -> None:
        """Store a message in the chronological history for a chat."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            history = self._messages.setdefault(chat_id, [])
            history.append(message)
            self._last_access[chat_id] = now

    async def get_history(self, chat_id: str) -> List[ConversationMessage]:
        """Return a copy of the stored message history for a chat."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            history = self._messages.get(chat_id, [])
            if history:
                self._last_access[chat_id] = now
            return list(history)

    async def get_model_history(self, chat_id: str) -> List[ModelMessage]:
        """Return the cached model-level history for a chat."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            history = self._model_messages.get(chat_id, [])
            if history:
                self._last_access[chat_id] = now
            return list(history)

    async def replace_model_history(
        self, chat_id: str, model_messages: List[ModelMessage]
    ) -> None:
        """Replace the stored model messages for a chat with a new snapshot."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            self._model_messages[chat_id] = list(model_messages)
            self._last_access[chat_id] = now

    async def reset(self, chat_id: str) -> None:
        """Forget the cached routing choice and message history for a chat."""

        async with self._lock:
            self._routes.pop(chat_id, None)
            self._messages.pop(chat_id, None)
            self._model_messages.pop(chat_id, None)
            self._last_access.pop(chat_id, None)

    async def reset_all(self) -> None:
        """Remove all cached routing decisions and histories."""

        async with self._lock:
            self._routes.clear()
            self._messages.clear()
            self._model_messages.clear()
            self._last_access.clear()

    def _purge_expired(self, now: float) -> None:
        """Drop cached entries that have been idle beyond the TTL."""

        expired = [
            chat_id
            for chat_id, last_access in self._last_access.items()
            if now - last_access > _CACHE_TTL_SECONDS
        ]
        for chat_id in expired:
            self._routes.pop(chat_id, None)
            self._messages.pop(chat_id, None)
            self._model_messages.pop(chat_id, None)
            self._last_access.pop(chat_id, None)


conversation_store = ConversationStore()


__all__ = ["ConversationMessage", "ConversationStore", "conversation_store"]
