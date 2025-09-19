"""Protocol guard utilities shared by the HTTP layer."""
from __future__ import annotations

from typing import Optional

from .api.models import ChatResponse


def apply_protocol_guards(content: str) -> Optional[ChatResponse]:
    """Handle protocol sanity commands without invoking the agent."""

    stripped = content.strip()
    if stripped == "ping":
        return ChatResponse(message="pong", base_random_keys=None, member_random_keys=None)
    if stripped.lower().startswith("return base random key:"):
        key = stripped.split(":", 1)[1].strip()
        return ChatResponse(message=None, base_random_keys=[key], member_random_keys=None)
    if stripped.lower().startswith("return member random key:"):
        key = stripped.split(":", 1)[1].strip()
        return ChatResponse(message=None, base_random_keys=None, member_random_keys=[key])
    return None
