"""In-memory state containers for routing and chat history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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


class ConversationStore:
    """Tracks router decisions and message history by chat identifier."""

    def __init__(self) -> None:
        self._routes: Dict[str, RouterRoute] = {}
        self._messages: Dict[str, List[ConversationMessage]] = {}
        self._model_messages: Dict[str, List[ModelMessage]] = {}
        self._lock = asyncio.Lock()

    async def get_route(self, chat_id: str) -> RouterRoute | None:
        """Return the cached routing decision for a chat if available."""

        async with self._lock:
            return self._routes.get(chat_id)

    async def remember_route(self, chat_id: str, route: RouterRoute) -> None:
        """Persist the routing decision for subsequent turns."""

        async with self._lock:
            self._routes[chat_id] = route

    async def append_message(
        self, chat_id: str, message: ConversationMessage
    ) -> None:
        """Store a message in the chronological history for a chat."""

        async with self._lock:
            history = self._messages.setdefault(chat_id, [])
            history.append(message)

    async def get_history(self, chat_id: str) -> List[ConversationMessage]:
        """Return a copy of the stored message history for a chat."""

        async with self._lock:
            history = self._messages.get(chat_id, [])
            return list(history)

    async def get_model_history(self, chat_id: str) -> List[ModelMessage]:
        """Return the cached model-level history for a chat."""

        async with self._lock:
            history = self._model_messages.get(chat_id, [])
            return list(history)

    async def replace_model_history(
        self, chat_id: str, model_messages: List[ModelMessage]
    ) -> None:
        """Replace the stored model messages for a chat with a new snapshot."""

        async with self._lock:
            self._model_messages[chat_id] = list(model_messages)

    async def reset(self, chat_id: str) -> None:
        """Forget the cached routing choice and message history for a chat."""

        async with self._lock:
            self._routes.pop(chat_id, None)
            self._messages.pop(chat_id, None)
            self._model_messages.pop(chat_id, None)

    async def reset_all(self) -> None:
        """Remove all cached routing decisions and histories."""

        async with self._lock:
            self._routes.clear()
            self._messages.clear()
            self._model_messages.clear()


conversation_store = ConversationStore()


__all__ = ["ConversationMessage", "ConversationStore", "conversation_store"]
