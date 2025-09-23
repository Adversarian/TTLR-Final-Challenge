"""Factory for constructing the multi-turn shopping assistant."""

from __future__ import annotations

import os
from threading import Lock

from cachetools import TTLCache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.config import CACHE_TTL_SECONDS

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from ..schemas import AgentReply
from ..tools import (
    FEATURE_LIST_FOR_BASES_TOOL,
    PRODUCT_SEARCH_WITH_FEATURES_TOOL,
    SELLER_CANDIDATE_SUMMARY_TOOL,
    SELLER_OFFERS_TOOL,
)
from .prompts import MULTI_TURN_SYSTEM_PROMPT


_AGENT_CACHE_KEY = "multi_turn"
_agent_cache: TTLCache[str, Agent[AgentDependencies, AgentReply]] = TTLCache(
    maxsize=1, ttl=CACHE_TTL_SECONDS
)
_agent_lock = Lock()


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
            PRODUCT_SEARCH_WITH_FEATURES_TOOL,
            FEATURE_LIST_FOR_BASES_TOOL,
            SELLER_CANDIDATE_SUMMARY_TOOL,
            SELLER_OFFERS_TOOL,
        ],
        instrument=InstrumentationSettings(),
        name="multi-turn-shopping-assistant",
    )


def get_multi_turn_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a cached multi-turn agent refreshed after idle expiration."""

    with _agent_lock:
        agent = _agent_cache.get(_AGENT_CACHE_KEY)
        if agent is None:
            agent = _build_multi_turn_agent()
        _agent_cache[_AGENT_CACHE_KEY] = agent
        return agent


__all__ = ["get_multi_turn_agent"]
