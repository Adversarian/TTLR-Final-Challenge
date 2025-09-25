"""Lightweight conversation memory helpers for the multi-turn flow."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

from pydantic import BaseModel, Field
from .contracts import TurnState


_MAX_SUMMARY_LENGTH = 200


class MemoryRecord(BaseModel):
    """Compact representation of the state persisted between turns."""

    state: TurnState = Field(..., description="Canonical turn state snapshot.")
    summary: str | None = Field(
        None,
        description=(
            "Short free-form summary of the exchange to prime the agent without"
            " replaying the full transcript."
        ),
    )

    def to_payload(self) -> dict:
        """Serialise the record into a JSON-friendly dictionary."""

        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MemoryRecord":
        """Hydrate a record from stored data."""

        return cls.model_validate(payload)


class ConversationMemory:
    """In-memory implementation of the conversation store."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, MemoryRecord] = {}

    def remember(
        self,
        chat_id: str,
        state: TurnState,
        *,
        summary: str | None = None,
        agent_messages: Sequence[str] | None = None,
    ) -> MemoryRecord:
        """Persist the state for the provided chat identifier."""

        summary_text = _truncate_summary(summary or _summarise_messages(agent_messages))
        record = MemoryRecord(state=state, summary=summary_text)
        self._records[chat_id] = record
        return record

    def recall(self, chat_id: str) -> MemoryRecord | None:
        """Return the stored record for the provided chat identifier."""

        record = self._records.get(chat_id)
        return record

    def forget(self, chat_id: str) -> None:
        """Remove stored state for the chat, if any."""

        if chat_id in self._records:
            del self._records[chat_id]

    def export(self, chat_id: str) -> dict | None:
        """Return a serialised snapshot suitable for persistence."""

        record = self._records.get(chat_id)
        if not record:
            return None
        return record.to_payload()

    def import_state(self, chat_id: str, payload: Mapping[str, object]) -> MemoryRecord:
        """Store a snapshot previously produced by :meth:`export`."""

        record = MemoryRecord.from_payload(payload)
        self._records[chat_id] = record
        return record


def _truncate_summary(summary: str | None) -> str | None:
    """Return a summary trimmed to the configured maximum length."""

    if not summary:
        return None
    summary = " ".join(summary.split())
    if len(summary) <= _MAX_SUMMARY_LENGTH:
        return summary
    return summary[: _MAX_SUMMARY_LENGTH - 1].rstrip() + "\u2026"


def _summarise_messages(messages: Sequence[str] | None) -> str | None:
    """Build a short summary from a list of agent messages."""

    if not messages:
        return None
    meaningful = [segment.strip() for segment in messages if segment and segment.strip()]
    if not meaningful:
        return None
    return " | ".join(meaningful)


__all__ = ["ConversationMemory", "MemoryRecord"]
