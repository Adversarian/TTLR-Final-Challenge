"""Factory for constructing the shopping assistant agent."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .dependencies import AgentDependencies
from .logging import _ensure_logfire
from .prompts import SYSTEM_PROMPT
from .schemas import AgentReply
from .tools import (
    FEATURE_LOOKUP_TOOL,
    PRODUCT_SEARCH_TOOL,
    SELLER_STATISTICS_TOOL,
)


@lru_cache(maxsize=1)
def get_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a configured agent instance with tool and logging support."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        settings=OpenAIChatModelSettings(
            temperature=0.1, parallel_tool_calls=True, openai_service_tier="priority"
        ),
    )

    return Agent(
        model=model,
        output_type=AgentReply,
        instructions=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[PRODUCT_SEARCH_TOOL, FEATURE_LOOKUP_TOOL, SELLER_STATISTICS_TOOL],
        instrument=InstrumentationSettings(),
        name="shopping-assistant",
    )


__all__ = ["get_agent"]
