"""Shared dependency container passed to agent runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..memory import ChatState
from ..db import DatabasePool


@dataclass
class AgentDependencies:
    """Objects made available to tools through the run context."""

    chat_id: str
    database: DatabasePool
    state: ChatState
