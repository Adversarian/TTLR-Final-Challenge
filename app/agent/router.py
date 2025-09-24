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
- single_turn → the user already references identifiable catalogue products or
  very specific questions that a single reply can cover. Typical examples are:
  • asking for a known product by name/code/sku,
  • requesting one attribute of a specific product,
  • comparing two (or more) explicitly named products,
  • asking for seller metrics (price, warranty, score) of a known base
    product.
- multi_turn → the user only names a broad product category or vague intent and
  needs guidance to pick a specific catalogue item or seller (e.g. "help me
  find a بخور", "چه گزینه‌هایی موجوده" with price/colour/material preferences
  but no concrete model). These are the exploratory shopping queries that
  demand several clarifying questions before recommending one seller.

Err toward single_turn unless the message is clearly ambiguous enough that the
assistant must gather more detail. Never send direct product-level requests to
multi_turn.

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
