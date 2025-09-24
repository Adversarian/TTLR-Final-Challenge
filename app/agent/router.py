"""Routing helper that classifies user requests by interaction depth."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings


class RouterDecision(BaseModel):
    """Structured output that captures the router verdict."""

    route: Literal["single_turn", "multi_turn"]


ROUTER_PROMPT = (
    "You are a fast routing module for a shopping assistant. Decide whether the"
    " latest user utterance can be satisfied with a single response or requires"
    " a short discovery dialogue before recommending a seller or product."
    "\n\n"
    "Return `multi_turn` when the user is still exploring a broad need, such as"
    " asking for help finding a suitable product or seller without naming a"
    " unique catalogue item. Messages that only mention a product type (e.g."
    " fridge, humidifier, bedsheet) or include loose constraints like price"
    " ranges, colours, preferred qualities, or desired room usage are"
    " considered ambiguous and therefore require multi-turn follow-up to gather"
    " the exact specification.\n"
    "Return `single_turn` when the user already pinpoints concrete catalogue"
    " entities or deterministic facts: explicit product names or model codes,"
    " requests for a specific attribute of a known product, seller statistics"
    " for an identified base item, direct price/score questions, or side-by-side"
    " comparisons between named products. In these cases the assistant should"
    " respond immediately without extra clarification.\n"
    "If the utterance mixes both signals, favour `multi_turn` only when the"
    " product reference is too vague to map to a single catalogue item."
    " Otherwise default to `single_turn`.\n\n"
    "Respond strictly as JSON with the shape {\"route\": \"single_turn\"} or"
    " {\"route\": \"multi_turn\"}. Do not include explanations or any extra"
    " keys."
)


@lru_cache(maxsize=1)
def get_router() -> Agent[None, RouterDecision]:
    """Construct and cache the lightweight interaction router."""

    model_name = os.getenv("OPENAI_ROUTER_MODEL") or os.getenv(
        "OPENAI_MODEL", "gpt-4o-mini"
    )
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        settings=ModelSettings(temperature=0, parallel_tool_calls=False),
    )

    return Agent(
        model=model,
        output_type=RouterDecision,
        instructions=ROUTER_PROMPT,
        name="interaction-router",
    )


__all__ = ["RouterDecision", "get_router"]

