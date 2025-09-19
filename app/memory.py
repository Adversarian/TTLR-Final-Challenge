"""In-memory conversation and context cache."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TTL_SECONDS = 15 * 60


@dataclass
class ChatMessageRecord:
    """Represents a single entry in the conversation history."""

    role: str
    content: str
    message_type: str = "text"


@dataclass
class ChatState:
    """Tracks memory associated with a chat session."""

    history: List[ChatMessageRecord] = field(default_factory=list)
    last_base_random_key: Optional[str] = None
    last_query: Optional[str] = None
    updated_at: float = field(default_factory=lambda: time.time())

    def touch(self) -> None:
        """Refresh the last-updated timestamp."""

        self.updated_at = time.time()


class ChatMemory:
    """A TTL-based memory cache keyed by chat id."""

    def __init__(self, ttl_seconds: int = TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, ChatState] = {}

    def get(self, chat_id: str) -> ChatState:
        """Return the state for a chat, initialising it when necessary."""

        state = self._store.get(chat_id)
        now = time.time()
        if state is None or now - state.updated_at > self._ttl:
            state = ChatState()
            self._store[chat_id] = state
        return state

    def update_base(self, chat_id: str, base_random_key: Optional[str], query: Optional[str]) -> None:
        """Persist the latest base resolution for the chat session."""

        state = self.get(chat_id)
        state.last_base_random_key = base_random_key
        if query:
            state.last_query = query
        state.touch()

    def append_history(self, chat_id: str, role: str, content: str, message_type: str = "text") -> None:
        """Add a message to the session history."""

        state = self.get(chat_id)
        state.history.append(ChatMessageRecord(role=role, content=content, message_type=message_type))
        state.touch()

    def clear(self, chat_id: str) -> None:
        """Remove cached state for a chat."""

        self._store.pop(chat_id, None)
