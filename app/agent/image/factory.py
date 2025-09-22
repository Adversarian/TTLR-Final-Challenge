"""Factory for constructing the multimodal vision agent."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from ..schemas import AgentReply
from .prompts import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_image_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a configured agent instance for image-centric conversations."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=ModelSettings(temperature=0.1, parallel_tool_calls=False),
    )

    return Agent(
        model=model,
        output_type=AgentReply,
        instructions=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[],
        instrument=InstrumentationSettings(),
        name="vision-shopping-assistant",
    )


__all__ = ["get_image_agent"]
