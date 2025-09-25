"""Factory for constructing the chat routing agent."""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..logging import _ensure_logfire
from .prompts import SYSTEM_PROMPT
from .schemas import RoutingDecision


@lru_cache(maxsize=1)
def get_router_agent() -> Agent[RoutingDecision]:
    """Return a configured agent that classifies requests by turn style."""

    _ensure_logfire()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key and not base_url:
        class _SingleTurnRouter:
            async def run(self, *args, **kwargs):  # pragma: no cover - trivial
                return SimpleNamespace(output=RoutingDecision(mode="single_turn"))

        return _SingleTurnRouter()

    model_name = os.getenv("OPENAI_ROUTER_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
        settings=ModelSettings(temperature=0.0, parallel_tool_calls=False),
    )

    return Agent(
        model=model,
        output_type=RoutingDecision,
        instructions=SYSTEM_PROMPT,
        instrument=InstrumentationSettings(),
        name="shopping-turn-router",
    )


__all__ = ["get_router_agent"]
