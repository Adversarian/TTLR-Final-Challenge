"""Agent factory utilities for the scenario 4 conversation graph."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from ..tools import PRODUCT_SEARCH_TOOL
from .schemas import (
    CandidatePresentation,
    ClarificationPlan,
    ConstraintExtraction,
    MemberFilterResponse,
    ProductFilterResponse,
    ResolutionSummary,
)
from .tools import (
    CATEGORY_FEATURE_STATISTICS_TOOL,
    FILTER_BASE_PRODUCTS_TOOL,
    FILTER_MEMBERS_TOOL,
)


def _build_model(temperature: float = 0.0, *, parallel_tools: bool = False) -> OpenAIChatModel:
    """Construct a chat model configured for the multi-turn agents."""

    _ensure_logfire()
    model_name = os.getenv(
        "OPENAI_MULTI_TURN_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        settings=ModelSettings(temperature=temperature, parallel_tool_calls=parallel_tools),
    )


CONSTRAINT_EXTRACTOR_PROMPT = """
You read the customer's most recent reply and distil it into structured constraints
for a shopping journey. Summarise what they want, capture category hints, price
ranges, brand mentions, required or forbidden attributes, and seller expectations
like warranty, shop rating, or city preferences. When they explicitly say a topic
does not matter, record the canonical token (brand, warranty, shop_score, city,
price, features) inside `dismissed_aspects`. Convert Persian numerals to digits
when recording numeric ranges. Stay concise and avoid guessing facts the user did
not state explicitly.
""".strip()


CLARIFICATION_PROMPT = """
You decide the next move in a multi-turn shopping dialogue. Consider the
collected constraints, questions already asked, candidate counts, dismissed
aspects, and remaining turns. Output one of five actions: ask_question,
search_products, present_candidates, resolve_members, or finalize. When asking
another question, respond with a single message (one or two sentences). Early in
the dialogue you may bundle multiple aspects together (مثلاً دسته، کاربرد و بودجه)
to gather context quickly, then shift to a focused follow-up (key feature,
preferred brand, price clarification) as details accumulate. Never revisit topics
listed in `dismissed_aspects`. When you already have two or more candidates and
at least two turns remain, present them for user selection instead of
re-querying. If the constraints look sufficient or the turn budget is nearly
exhausted, advance to searching or resolving members.
""".strip()


SEARCH_PROMPT = """
You translate the aggregated constraints into actual catalogue candidates. Call
`filter_base_products_by_constraints` once with the strongest available inputs
while skipping dimensions the user dismissed. Optionally inspect category-wide
features with `category_feature_statistics` if it helps explain which attributes
matter. Return the tool output unchanged. Do not invent catalogue data or retry
with identical arguments.
""".strip()


CANDIDATE_REDUCER_PROMPT = """
When multiple base products remain, prepare a short comparison to help the user
choose. Mention at most three leading options, highlighting the key differences
(e.g., capacity, material, included accessories, standout feature matches) and
end with a question asking which one fits best. Keep the tone polite and avoid
overwhelming detail.
""".strip()


MEMBER_RESOLVER_PROMPT = """
Retrieve member (shop) offers for the resolved base product. Use
`filter_members_by_constraints` exactly once. Prefer the cheapest option that
meets warranty, score, and city requirements unless the user dismissed that
dimension or the prompt indicates forced finalisation. In that case, gracefully
relax the optional filters but still return the best available member. Do not
fabricate member keys or repeat the tool call with the same arguments.
""".strip()


FINALISER_PROMPT = """
Provide the final recommendation once a single member offer is chosen. Mention
the decisive factors (price, warranty, score, feature alignment) in one or two
sentences and return the member_random_key exactly once. No extra keys or
follow-up questions are allowed.
""".strip()


@lru_cache(maxsize=1)
def get_constraint_extractor_agent() -> Agent[AgentDependencies, ConstraintExtraction]:
    """Return the agent responsible for extracting structured constraints."""

    return Agent(
        model=_build_model(temperature=0.0),
        output_type=ConstraintExtraction,
        instructions=CONSTRAINT_EXTRACTOR_PROMPT,
        deps_type=AgentDependencies,
        instrument=InstrumentationSettings(),
        name="scenario4-constraint-extractor",
    )


@lru_cache(maxsize=1)
def get_clarification_agent() -> Agent[AgentDependencies, ClarificationPlan]:
    """Return the planner agent that decides the next dialogue action."""

    return Agent(
        model=_build_model(temperature=0.1),
        output_type=ClarificationPlan,
        instructions=CLARIFICATION_PROMPT,
        deps_type=AgentDependencies,
        instrument=InstrumentationSettings(),
        name="scenario4-clarification-planner",
    )


@lru_cache(maxsize=1)
def get_search_agent() -> Agent[AgentDependencies, ProductFilterResponse]:
    """Return the agent that maps constraints to catalogue candidates."""

    return Agent(
        model=_build_model(temperature=0.1, parallel_tools=True),
        output_type=ProductFilterResponse,
        instructions=SEARCH_PROMPT,
        deps_type=AgentDependencies,
        tools=[PRODUCT_SEARCH_TOOL, CATEGORY_FEATURE_STATISTICS_TOOL, FILTER_BASE_PRODUCTS_TOOL],
        instrument=InstrumentationSettings(),
        name="scenario4-catalogue-searcher",
    )


@lru_cache(maxsize=1)
def get_candidate_reducer_agent() -> Agent[AgentDependencies, CandidatePresentation]:
    """Return the agent that phrases comparison questions for the user."""

    return Agent(
        model=_build_model(temperature=0.2),
        output_type=CandidatePresentation,
        instructions=CANDIDATE_REDUCER_PROMPT,
        deps_type=AgentDependencies,
        instrument=InstrumentationSettings(),
        name="scenario4-candidate-reducer",
    )


@lru_cache(maxsize=1)
def get_member_resolver_agent() -> Agent[AgentDependencies, MemberFilterResponse]:
    """Return the agent that filters shop offers for the chosen base product."""

    return Agent(
        model=_build_model(temperature=0.0, parallel_tools=True),
        output_type=MemberFilterResponse,
        instructions=MEMBER_RESOLVER_PROMPT,
        deps_type=AgentDependencies,
        tools=[FILTER_MEMBERS_TOOL],
        instrument=InstrumentationSettings(),
        name="scenario4-member-resolver",
    )


@lru_cache(maxsize=1)
def get_finaliser_agent() -> Agent[AgentDependencies, ResolutionSummary]:
    """Return the agent that crafts the final recommendation."""

    return Agent(
        model=_build_model(temperature=0.0),
        output_type=ResolutionSummary,
        instructions=FINALISER_PROMPT,
        deps_type=AgentDependencies,
        instrument=InstrumentationSettings(),
        name="scenario4-finaliser",
    )


__all__ = [
    "get_candidate_reducer_agent",
    "get_clarification_agent",
    "get_constraint_extractor_agent",
    "get_finaliser_agent",
    "get_member_resolver_agent",
    "get_search_agent",
]

