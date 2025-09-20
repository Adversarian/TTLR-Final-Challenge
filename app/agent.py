"""Agent setup and tool definitions for the shopping assistant."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import List, Sequence

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.tools import RunContext, Tool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import BaseProduct


@dataclass
class AgentDependencies:
    """Runtime dependencies passed to the agent on every run."""

    session: AsyncSession


class ProductMatch(BaseModel):
    """Represents a single product candidate returned by a search."""

    random_key: str = Field(..., description="The base product random key.")
    persian_name: str = Field(..., description="Canonical Persian name of the product.")
    english_name: str | None = Field(
        None, description="English product name when available."
    )
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score relative to the input query."
    )


class ProductSearchResult(BaseModel):
    """Collection of ranked product matches for a customer request."""

    query: str = Field(..., description="Normalized query that was searched.")
    matches: List[ProductMatch] = Field(
        default_factory=list,
        description="Ranked list of the strongest matches (best first).",
    )


class FeatureLookupResult(BaseModel):
    """Best effort lookup for a specific product attribute."""

    base_random_key: str = Field(..., description="Target product random key.")
    feature_name: str | None = Field(
        None, description="Feature name that best matches the request."
    )
    value: str | None = Field(
        None, description="Human readable value for the matched feature."
    )
    available_features: List[str] = Field(
        default_factory=list,
        description="All feature labels discovered for the product.",
    )


class AgentReply(BaseModel):
    """Structured response emitted by the agent."""

    message: str | None = Field(
        None, description="Assistant message to display to the customer."
    )
    base_random_keys: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Base product keys relevant to the response (at most 10).",
    )
    member_random_keys: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Member product keys relevant to the response (at most 10).",
    )

    def clipped(self) -> "AgentReply":
        """Return a copy trimmed to the API list length limits."""

        return AgentReply(
            message=self.message,
            base_random_keys=self.base_random_keys[:10],
            member_random_keys=self.member_random_keys[:10],
        )


def _configure_logfire() -> None:
    """Configure Logfire instrumentation if it has not been configured yet."""

    token = os.getenv("LOGFIRE_API_KEY")
    if token:
        logfire.configure(token=token)
    else:
        logfire.configure()


_LOGFIRE_READY = False


def _ensure_logfire() -> None:
    global _LOGFIRE_READY
    if not _LOGFIRE_READY:
        _configure_logfire()
        _LOGFIRE_READY = True


def _normalize_text(value: str) -> str:
    """Return a lightly normalised version of Persian/English text."""

    replacements = {
        "\u064a": "ی",  # Arabic Yeh -> Persian Yeh
        "\u0643": "ک",  # Arabic Kaf -> Persian Keheh
        "\u06cc": "ی",  # Farsi Yeh variant
        "\u06a9": "ک",  # Keheh variant
    }
    lowered = value.strip().lower()
    for src, dest in replacements.items():
        lowered = lowered.replace(src, dest)
    return lowered


async def _fetch_top_matches(
    session: AsyncSession, normalized_query: str, limit: int = 10
) -> Sequence[ProductMatch]:
    """Return the strongest matches for the provided search query."""

    if not normalized_query:
        return []

    similarity_name = func.greatest(
        func.similarity(BaseProduct.persian_name, normalized_query),
        func.similarity(func.coalesce(BaseProduct.english_name, ""), normalized_query),
    )

    score = similarity_name.label("score")

    stmt = (
        select(
            BaseProduct.random_key,
            BaseProduct.persian_name,
            BaseProduct.english_name,
            score,
        )
        .order_by(score.desc())
        .where(similarity_name > 0.0)
        .limit(limit)
    )

    result = await session.execute(stmt)
    matches: List[ProductMatch] = []
    for random_key, persian_name, english_name, score in result:
        matches.append(
            ProductMatch(
                random_key=random_key,
                persian_name=persian_name,
                english_name=english_name,
                similarity=float(score or 0.0),
            )
        )
    return matches


async def _find_product_by_key(
    session: AsyncSession, raw_query: str
) -> ProductMatch | None:
    """Return a direct random-key match when one exists."""

    trimmed = raw_query.strip()
    if not trimmed:
        return None

    product = await session.get(BaseProduct, trimmed)
    if product is None:
        return None

    return ProductMatch(
        random_key=product.random_key,
        persian_name=product.persian_name,
        english_name=product.english_name,
        similarity=1.0,
    )


def _flatten_features(extra_features: dict | None) -> List[tuple[str, str]]:
    """Flatten a nested JSON blob into simple feature/value pairs."""

    if not extra_features:
        return []

    flattened: List[tuple[str, str]] = []

    def _walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                new_prefix = f"{prefix} {key}".strip()
                _walk(new_prefix, nested)
        elif isinstance(value, list):
            str_value = ", ".join(str(item) for item in value)
            flattened.append((prefix, str_value))
        else:
            flattened.append((prefix, str(value)))

    for key, value in extra_features.items():
        _walk(key, value)

    return flattened


def _lookup_feature(
    features: Sequence[tuple[str, str]], normalized_query: str
) -> tuple[str | None, str | None]:
    """Select the feature that most closely matches the query."""

    if not features:
        return None, None

    from difflib import SequenceMatcher

    best_name: str | None = None
    best_value: str | None = None
    best_score = 0.0

    for name, value in features:
        score = SequenceMatcher(
            None, _normalize_text(name), normalized_query
        ).ratio()
        if score > best_score:
            best_score = score
            best_name = name
            best_value = value

    return best_name, best_value


async def _search_base_products(
    ctx: RunContext[AgentDependencies], query: str
) -> ProductSearchResult:
    """Resolve a customer request to likely base products."""

    normalized = _normalize_text(query)
    session = ctx.deps.session

    direct_match = await _find_product_by_key(session, query)
    if direct_match is not None:
        return ProductSearchResult(query=normalized, matches=[direct_match])

    matches = await _fetch_top_matches(session, normalized)
    return ProductSearchResult(query=normalized, matches=list(matches))


async def _fetch_feature_details(
    ctx: RunContext[AgentDependencies],
    base_random_key: str,
    feature_query: str,
) -> FeatureLookupResult:
    """Retrieve descriptive attributes for the requested base product."""

    normalized_feature = _normalize_text(feature_query)
    normalized_key = _normalize_text(base_random_key)
    session = ctx.deps.session

    trimmed_key = base_random_key.strip()
    product = await session.get(BaseProduct, trimmed_key) if trimmed_key else None

    if product is not None:
        extra_features = product.extra_features
    else:
        stmt = select(BaseProduct.extra_features).where(
            func.lower(BaseProduct.random_key) == normalized_key
        )
        result = await session.execute(stmt)
        extra_features = result.scalar_one_or_none()

    flattened = _flatten_features(
        extra_features if isinstance(extra_features, dict) else {}
    )
    matched_name, matched_value = _lookup_feature(flattened, normalized_feature)

    return FeatureLookupResult(
        base_random_key=base_random_key,
        feature_name=matched_name,
        value=matched_value,
        available_features=[name for name, _ in flattened],
    )


PRODUCT_SEARCH_TOOL = Tool(
    _search_base_products,
    name="search_base_products",
    description=(
        "Use this tool to map the customer's language to actual base products. "
        "Your search string MUST be an exact substring from the latest user message. "
        "Provide that snippet from the customer request so the tool can run a fuzzy lookup. "
        "It returns up to 10 of the strongest catalogue matches (best first) with random keys and similarity scores."
    ),
)


FEATURE_LOOKUP_TOOL = Tool(
    _fetch_feature_details,
    name="get_product_feature",
    description=(
        "Use this tool after you know which base product the user means. "
        "Give it the base product random key and the specific feature text you need. "
        "It returns the closest feature label, its value, and all available feature names."
    ),
)


SYSTEM_PROMPT = (
    "You are a concise but helpful shopping assistant. Ground every answer in the product catalogue by identifying the most relevant base product before making recommendations or quoting attributes.\n\n"
    "SCENARIO GUIDE:\n"
    "- Product procurement requests: resolve the customer's wording to a single catalogue item and answer in one turn with the best-matching base random key.\n"
    "- Feature clarification requests: locate the product first, then surface the requested attribute's value directly without asking for more details.\n"
    "- Connectivity pings or other simple sanity checks may be answered with the static shortcuts provided by the API layer.\n\n"
    "GENERAL PRINCIPLES:\n"
    "- Answer deterministic product or feature questions in a single turn; do not ask clarifying questions even if confidence is modest—pick the strongest match and, if necessary, acknowledge uncertainty succinctly.\n"
    "- Always ground statements in actual catalogue data and keep explanations brief, factual, and free of invented information.\n"
    "- Only include product keys when they are explicitly required or necessary for the response, keeping lists trimmed to at most one base key by default.\n\n"
    "TOOL USAGE:\n"
    "- search_base_products: Call this whenever you need to resolve what product the user references. The search string MUST be an exact substring of the user's latest message. Review up to ten returned matches and choose the option whose identifiers appear verbatim in the request.\n"
    "- get_product_feature: After identifying the product, use this to retrieve catalogue features. Provide the base random key and the exact feature wording from the user to receive the closest label, its value, and all available feature names for additional context.\n"
)


@lru_cache(maxsize=1)
def get_agent() -> Agent[AgentDependencies, AgentReply]:
    """Return a configured agent instance with tool and logging support."""

    _ensure_logfire()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model = OpenAIChatModel(model_name, api_key=os.getenv("OPENAI_API_KEY"))

    return Agent(
        model=model,
        output_type=AgentReply,
        instructions=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[PRODUCT_SEARCH_TOOL, FEATURE_LOOKUP_TOOL],
        instrument=InstrumentationSettings(),
        name="shopping-assistant",
    )
