"""Constraint extraction helper for the multi-turn workflow."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Mapping

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from .models import ConstraintUpdate, MemberFilters, MultiTurnState


_PARSER_PROMPT = (
    "You are an extraction helper assisting a shopping assistant that must find"
    " the exact shop+product (member) the customer wants. Read the latest"
    " customer message together with the conversation state and return a JSON"
    " object describing any newly inferred filters.\n\n"
    "Guidelines:\n"
    "- Prefer broad categories over guesses when the user is unsure.\n"
    "- Interpret Persian numbers, ranges, and price units. Prices are in تومان"
    " unless explicitly stated otherwise.\n"
    "- When the user refuses to specify a field (e.g. 'برند مهم نیست'), include"
    " that field name inside `excluded_fields`.\n"
    "- Detect explicit shop selections (shop ids, member keys, or option numbers)"
    " and populate `selected_member_random_key` when unambiguous.\n"
    "- Mark `rejected_candidates` as true when the user indicates that none of"
    " the previously suggested members are acceptable.\n"
    "- Keep `text_queries` concise keywords that can help fuzzy text search"
    " (product type, standout features, capacities, colours, model codes).\n"
    "- Only fill numeric fields when the user provides a concrete value or"
    " range.\n"
    "- Return an empty list or null for fields you cannot update.\n\n"
    "Output strictly as JSON matching the provided schema without additional"
    " commentary."
)


def _serialise_filters(filters: MemberFilters) -> Mapping[str, Any]:
    """Return a JSON-friendly snapshot of the collected filters."""

    return {
        "text_queries": filters.text_queries,
        "category_id": filters.category_id,
        "brand_id": filters.brand_id,
        "city_id": filters.city_id,
        "min_price": filters.min_price,
        "max_price": filters.max_price,
        "requires_warranty": filters.requires_warranty,
        "min_score": filters.min_score,
        "max_score": filters.max_score,
        "preferred_shop_ids": filters.preferred_shop_ids,
        "allowed_shop_ids": filters.allowed_shop_ids,
        "excluded_fields": sorted(filters.excluded_fields),
        "asked_questions": sorted(filters.asked_questions),
    }


@lru_cache(maxsize=1)
def _build_agent() -> Agent[None, ConstraintUpdate]:
    """Instantiate the lightweight parser agent."""

    model_name = (
        os.getenv("OPENAI_MULTI_TURN_MODEL")
        or os.getenv("OPENAI_ROUTER_MODEL")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
        instructions=_PARSER_PROMPT,
        output_type=ConstraintUpdate,
        name="constraint-parser",
    )


async def parse_constraints(
    state: MultiTurnState,
    user_message: str,
) -> ConstraintUpdate:
    """Run the parser agent and persist its message history on the state."""

    filters_snapshot = json.dumps(
        _serialise_filters(state.filters), ensure_ascii=False, separators=(",", ":")
    )
    pending = state.pending_question_key or "none"
    prompt = (
        "CONVERSATION STATE:\n"
        f"pending_question: {pending}\n"
        f"filters: {filters_snapshot}\n"
        "USER_MESSAGE:\n"
        f"{user_message.strip()}"
    )

    agent = _build_agent()
    result = await agent.run(
        user_prompt=prompt,
        message_history=list(state.parser_history) or None,
        usage_limits=UsageLimits(request_limit=1, tool_calls_limit=0),
    )

    state.parser_history = result.all_messages()
    return result.output


__all__ = ["parse_constraints"]

