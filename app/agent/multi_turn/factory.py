"""Factory for constructing the multi-turn shopping assistant."""

from __future__ import annotations

import os
from threading import Lock
from time import monotonic

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from ..schemas import AgentReply
from ..tools import (
    FEATURE_LOOKUP_TOOL,
    PRODUCT_SEARCH_TOOL,
    SELLER_CANDIDATE_SUMMARY_TOOL,
    SELLER_OFFERS_TOOL,
    SELLER_STATISTICS_TOOL,
)
from .prompts import MULTI_TURN_SYSTEM_PROMPT


_CACHE_TTL_SECONDS = 120.0
_agent_lock = Lock()
_cached_agent: Agent[AgentDependencies, AgentReply] | None = None
_last_access: float = 0.0


def _build_multi_turn_agent() -> Agent[AgentDependencies, AgentReply]:
    """Construct a new multi-turn agent instance."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=ModelSettings(temperature=0.3, parallel_tool_calls=True),
    )

    return Agent(
        model=model,
        output_type=AgentReply,
        instructions=MULTI_TURN_SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[
            PRODUCT_SEARCH_TOOL,
            FEATURE_LOOKUP_TOOL,
            SELLER_CANDIDATE_SUMMARY_TOOL,
            SELLER_STATISTICS_TOOL,
            SELLER_OFFERS_TOOL,
        ],
        instrument=InstrumentationSettings(),
        name="multi-turn-shopping-assistant",
    )


def get_multi_turn_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a cached multi-turn agent refreshed after two idle minutes."""

    global _cached_agent, _last_access

    now = monotonic()
    with _agent_lock:
        if (
            _cached_agent is None
            or now - _last_access > _CACHE_TTL_SECONDS
        ):
            _cached_agent = _build_multi_turn_agent()
        _last_access = now
        return _cached_agent


__all__ = ["get_multi_turn_agent"]
