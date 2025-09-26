"""Factory helpers for constructing the multi-turn conversation agent."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from .prompts import MULTI_TURN_PROMPT
from .schemas import MultiTurnAgentReply
from .tools import SEARCH_MEMBERS_TOOL


@lru_cache(maxsize=1)
def get_multi_turn_agent() -> Agent[AgentDependencies, MultiTurnAgentReply]:
    """Return the configured multi-turn agent instance."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MULTI_TURN_MODEL", "gpt-4.1-mini")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=OpenAIChatModelSettings(
            temperature=0.1,
            parallel_tool_calls=False,
            # openai_service_tier="priority",
        ),
    )

    return Agent(
        model=model,
        output_type=MultiTurnAgentReply,
        instructions=MULTI_TURN_PROMPT,
        deps_type=AgentDependencies,
        tools=[SEARCH_MEMBERS_TOOL],
        instrument=InstrumentationSettings(),
        name="shopping-multi-turn",
    )


__all__ = ["get_multi_turn_agent"]
