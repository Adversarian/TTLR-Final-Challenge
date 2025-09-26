"""Factory for constructing the vision request router."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ..logging import _ensure_logfire
from .prompts import VISION_ROUTER_PROMPT
from .schemas import VisionRouteDecision


@lru_cache(maxsize=1)
def get_vision_router() -> Agent[None, VisionRouteDecision]:
    """Return an agent that decides how to handle vision queries."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=OpenAIChatModelSettings(
            temperature=0,
            parallel_tool_calls=False,
            openai_service_tier="priority",
        ),
    )

    return Agent(
        model=model,
        output_type=VisionRouteDecision,
        instructions=VISION_ROUTER_PROMPT,
        instrument=InstrumentationSettings(),
        name="vision-router",
    )


__all__ = ["get_vision_router"]
