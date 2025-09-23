"""Router agent that decides which assistant should handle a request."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

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
    reason: str | None = Field(
        None,
        description=(
            "Short natural-language justification for the chosen route."
        ),
    )


@lru_cache(maxsize=1)
def get_router_agent() -> Agent[None, RouterDecision]:
    """Return an agent that classifies whether a chat needs multiple turns."""

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


__all__ = ["RouterDecision", "RouterRoute", "get_router_agent"]
