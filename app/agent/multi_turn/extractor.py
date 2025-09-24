"""Constraint extraction agent for the multi-turn workflow."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..dependencies import AgentDependencies
from .schemas import ConstraintUpdate
from .state import MultiTurnSession


_EXTRACTION_PROMPT = (
    "You convert Persian shopper replies into structured filters for finding a"
    " precise shop+product member.\n"
    "Always produce valid JSON that matches the ConstraintUpdate schema."
    " Guidance:\n"
    "- Read the latest user message in the context of the provided state.\n"
    "- text_queries should include key nouns or model numbers that describe the"
    " product.\n"
    "- feature_hints capture loose attributes (material, capacity, colour).\n"
    "- Detect when the shopper refuses a field (e.g. می‌گوید برند مهم نیست) and"
    " add that field name to excluded_fields. Common field names: category_id,"
    " brand_id, city_id, price_range, requires_warranty, score.\n"
    "- When the shopper answers a multi-part question, fill as many fields as"
    " possible in one response.\n"
    "- Preferred shop selections must translate into preferred_shop_ids (list of"
    " integers) or selected_member_random_key when they reference our listed"
    " options.\n"
    "- clear_fields should include any field that must be reset to None based on"
    " the reply (e.g. user changes their mind).\n"
    "- Keep booleans explicit (True/False). Convert Persian numbers or ranges to"
    " integers. Use toman units (no ریال).\n"
    "- notes can summarise any non-structured insights for logging.\n"
    "Respond ONLY with JSON."
)


@lru_cache(maxsize=1)
def get_constraint_agent() -> Agent[AgentDependencies, ConstraintUpdate]:
    """Return the cached extraction agent."""

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        settings=ModelSettings(temperature=0, parallel_tool_calls=False),
    )

    return Agent(
        model=model,
        output_type=ConstraintUpdate,
        instructions=_EXTRACTION_PROMPT,
        deps_type=AgentDependencies,
        name="member-constraint-extractor",
    )


async def extract_constraints(
    session: MultiTurnSession,
    deps: AgentDependencies,
    message: str,
) -> ConstraintUpdate:
    """Run the extraction agent with persistent memory for the chat."""

    state_payload = session.state.to_prompt_payload()
    prompt = {
        "state": state_payload,
        "message": message.strip(),
    }
    formatted_prompt = json.dumps(prompt, ensure_ascii=False)

    result = await get_constraint_agent().run(
        user_prompt=formatted_prompt,
        deps=deps,
        message_history=session.constraint_history,
        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=0),
    )
    session.constraint_history = result.all_messages()
    return result.output
