"""Routing helper for deciding between single-turn and multi-turn handling."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

ROUTER_SYSTEM_PROMPT = """
You triage Persian shopping queries for an AI assistant. Choose whether the
request can be fulfilled in a single response or if it demands a multi-turn
dialogue to gather clarifying details.

Label guidance:
- single_turn → the user already names a specific catalogue product, model
  number, SKU, or directly asks for a factual attribute/seller metric about a
  known item. These are immediately answerable.
- multi_turn → the user only mentions a product category (e.g. لوستر، بخور،
  سرویس قابلمه) together with loose preferences such as price range, material,
  colour, shipping, or "help me find" phrasing. These require follow-up
  questions to identify the exact product before querying sellers.

Default to single_turn unless the request is clearly ambiguous enough that the
assistant must ask clarifying questions.

Respond with a JSON object containing a single field `decision` set to either
`single_turn` or `multi_turn`.
""".strip()


class RouterDecision(str, Enum):
    """Labels returned by the routing agent."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


class RouterReply(BaseModel):
    """Structured output from the routing agent."""

    decision: RouterDecision


@lru_cache(maxsize=1)
def get_router() -> Agent[None, RouterReply]:
    """Return a cached routing agent for classifying incoming prompts."""

    model_name = os.getenv(
        "OPENAI_ROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        settings=ModelSettings(temperature=0),
    )

    return Agent(
        model=model,
        output_type=RouterReply,
        instructions=ROUTER_SYSTEM_PROMPT,
        name="scenario-router",
    )


__all__ = ["RouterDecision", "RouterReply", "get_router"]
