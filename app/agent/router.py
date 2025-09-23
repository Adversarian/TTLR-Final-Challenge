"""Router agent that decides which assistant should handle a request."""

from __future__ import annotations

import os
from enum import Enum
from threading import Lock

from cachetools import TTLCache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.config import CACHE_TTL_SECONDS

from .logging import _ensure_logfire
from .prompts import ROUTER_SYSTEM_PROMPT


class RouterRoute(str, Enum):
    """Enumeration of the available routing targets."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


class RouterDecision(BaseModel):
    """Structured response produced by the router agent."""

    route: RouterRoute = Field(
        ..., description="Destination agent that should handle the request."
    )


_AGENT_CACHE_KEY = "router"
_router_cache: TTLCache[str, Agent[None, RouterDecision]] = TTLCache(
    maxsize=1, ttl=CACHE_TTL_SECONDS
)
_router_lock = Lock()


def _build_router_agent() -> Agent[None, RouterDecision]:
    """Construct a fresh router agent instance."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=ModelSettings(temperature=0.0, parallel_tool_calls=False),
    )

    return Agent(
        model=model,
        output_type=RouterDecision,
        instructions=ROUTER_SYSTEM_PROMPT,
        instrument=InstrumentationSettings(),
        name="shopping-router",
    )


def get_router_agent() -> Agent[None, RouterDecision]:
    """Return a cached router agent refreshed after idle expiration."""

    with _router_lock:
        agent = _router_cache.get(_AGENT_CACHE_KEY)
        if agent is None:
            agent = _build_router_agent()
        _router_cache[_AGENT_CACHE_KEY] = agent
        return agent


__all__ = ["RouterDecision", "RouterRoute", "get_router_agent"]
