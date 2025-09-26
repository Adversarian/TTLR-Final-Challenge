"""Factory for constructing the lightweight conversation router agent."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ..logging import _ensure_logfire
from .prompts import ROUTER_PROMPT
from .schemas import RouterDecision


@lru_cache(maxsize=1)
def get_conversation_router() -> Agent[None, RouterDecision]:
    """Return an agent that classifies chats as single or multi turn."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=OpenAIChatModelSettings(
            temperature=0, parallel_tool_calls=False, openai_service_tier="priority"
        ),
    )

    return Agent(
        model=model,
        output_type=RouterDecision,
        instructions=ROUTER_PROMPT,
        instrument=InstrumentationSettings(),
        name="conversation-router",
    )


__all__ = ["get_conversation_router"]
