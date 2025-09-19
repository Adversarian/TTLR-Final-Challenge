"""Agent factory wiring system prompt and tools together."""
from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from ..config import get_settings
from .context import AgentDependencies
from .models import AgentResponse
from .prompt import build_system_prompt
from .tools import TOOLKIT


@lru_cache(maxsize=1)
def get_agent() -> Agent[AgentDependencies, AgentResponse]:
    """Instantiate the shopping assistant agent once per process."""

    settings = get_settings()
    model_identifier = settings.primary_model
    agent = Agent[AgentDependencies, AgentResponse](
        model=model_identifier,
        instructions=build_system_prompt,
        output_type=AgentResponse,
        tools=TOOLKIT,
        model_settings=ModelSettings(temperature=0.0),
        name="shopping-assistant",
    )
    return agent
