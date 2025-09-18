"""Scenario-specific helpers for the assistant."""

from __future__ import annotations

import re
from typing import Optional

from app.models.chat import ChatResponse

_BASE_KEY_PATTERN = re.compile(r"return\s+base\s+random\s+key\s*:\s*(.+)", re.IGNORECASE)
_MEMBER_KEY_PATTERN = re.compile(
    r"return\s+member\s+random\s+key\s*:\s*(.+)", re.IGNORECASE
)


def handle_scenario_zero(message: str) -> Optional[ChatResponse]:
    cleaned = message.strip()
    if not cleaned:
        return None
    if cleaned.lower() == "ping":
        return ChatResponse(message="pong")

    base_match = _BASE_KEY_PATTERN.match(cleaned)
    if base_match:
        key = base_match.group(1).strip()
        if key:
            return ChatResponse(base_random_keys=[key])

    member_match = _MEMBER_KEY_PATTERN.match(cleaned)
    if member_match:
        key = member_match.group(1).strip()
        if key:
            return ChatResponse(member_random_keys=[key])
    return None


__all__ = ["handle_scenario_zero"]
