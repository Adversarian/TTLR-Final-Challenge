"""Pydantic schemas shared across the scenario 4 agent graph."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class FeatureConstraintModel(BaseModel):
    """Structured representation of a desired product feature."""

    name: str = Field(..., description="Canonical label for the feature key.")
    value: str = Field(..., description="Desired value or descriptor for the feature.")
    match: Literal["equals", "contains", "min_value", "max_value"] = Field(
        "contains",
        description=(
            "How the value should be matched. 'equals' requires an exact match,"
            " 'contains' looks for the value as a substring, while 'min_value'"
            " and 'max_value' describe numeric bounds when applicable."
        ),
    )


class ConstraintExtraction(BaseModel):
    """Information distilled from the user's latest message."""

    summary: str = Field(
        ...,
        description="Natural language recap of the user's needs to aid later prompts.",
    )
    category_hint: str | None = Field(
        None, description="User-stated category or product archetype if provided."
    )
    brand_preferences: list[str] = Field(
        default_factory=list,
        description="Any specific brands or brand adjectives the user mentioned.",
    )
    price_min: int | None = Field(
        None,
        ge=0,
        description="Lower bound of the user's acceptable price range in toman.",
    )
    price_max: int | None = Field(
        None,
        ge=0,
        description="Upper bound of the user's acceptable price range in toman.",
    )
    required_features: list[FeatureConstraintModel] = Field(
        default_factory=list,
        description="Must-have attributes that should be enforced during filtering.",
    )
    optional_features: list[FeatureConstraintModel] = Field(
        default_factory=list,
        description="Nice-to-have attributes that can help with tie-breaking.",
    )
    excluded_features: list[FeatureConstraintModel] = Field(
        default_factory=list,
        description="Attributes the user explicitly wants to avoid.",
    )
    city_preferences: list[str] = Field(
        default_factory=list,
        description="Cities requested for fulfilment or delivery if stated.",
    )
    require_warranty: bool | None = Field(
        None,
        description="Whether the user demanded shops with warranty-backed offers only.",
    )
    min_shop_score: float | None = Field(
        None,
        ge=0.0,
        le=5.0,
        description="Minimum acceptable shop rating on the Torob 0-5 scale.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Other salient descriptors or synonyms worth injecting into search.",
    )
    dismissed_aspects: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical names of shopping dimensions the user explicitly marked as"
            " irrelevant (e.g., brand, warranty, shop_score, city, price, features)."
        ),
    )


class ClarificationPlan(BaseModel):
    """Planner output describing how to progress the dialogue."""

    action: Literal[
        "ask_question",
        "search_products",
        "present_candidates",
        "resolve_members",
        "finalize",
    ] = Field(..., description="Next step for the coordinator to execute.")
    question: str | None = Field(
        None,
        description="Follow-up question for the customer when more detail is needed.",
    )
    rationale: str = Field(
        ...,
        description="Short justification that summarises the planner's reasoning.",
    )


class CategoryFeatureStatistic(BaseModel):
    """High-level frequency information about features within a category."""

    feature_path: str = Field(
        ...,
        description="Flattened JSON path (keys joined by spaces) for the feature.",
    )
    sample_values: list[str] = Field(
        default_factory=list,
        description="Representative values observed for the feature.",
    )
    occurrences: int = Field(
        ..., ge=0, description="Number of products in the sample containing this feature."
    )


class FilteredProduct(BaseModel):
    """Product candidate returned after applying structured constraints."""

    base_random_key: str = Field(..., description="Identifier of the base product.")
    persian_name: str = Field(..., description="Persian display name of the product.")
    english_name: str | None = Field(
        None, description="English name of the product when available."
    )
    category_title: str | None = Field(
        None, description="Category label for the product when joined."
    )
    brand_title: str | None = Field(
        None, description="Brand name for the product when joined."
    )
    min_price: int | None = Field(
        None, description="Cheapest observed price among member offers."
    )
    max_price: int | None = Field(
        None, description="Most expensive observed price among member offers."
    )
    matched_features: list[str] = Field(
        default_factory=list,
        description="Features that aligned with the collected constraints.",
    )
    match_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Relative confidence score derived from how well constraints matched.",
    )


class ProductFilterResponse(BaseModel):
    """Collection of candidate products returned by the filter tool."""

    candidates: list[FilteredProduct] = Field(
        default_factory=list,
        description="Ranked list of viable base products after filtering.",
    )


class MemberOffer(BaseModel):
    """Single shop listing for a resolved base product."""

    member_random_key: str = Field(
        ..., description="Identifier of the member (shop-specific) listing."
    )
    shop_id: int = Field(..., description="Identifier of the shop offering the product.")
    price: int = Field(..., ge=0, description="Price of the offer in toman.")
    has_warranty: bool = Field(
        ..., description="Whether the shop provides Torob warranty for this offer."
    )
    shop_score: Decimal | None = Field(
        None, description="Shop rating on Torob's 0-5 scale when available."
    )
    city_name: str | None = Field(
        None, description="City where the shop is registered, when available."
    )
    matched_constraints: list[str] = Field(
        default_factory=list,
        description="Human-readable notes describing which constraints this offer satisfies.",
    )
    match_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Relative ranking score derived from the satisfied constraints.",
    )


class MemberFilterResponse(BaseModel):
    """Response payload returned after filtering member offers."""

    offers: list[MemberOffer] = Field(
        default_factory=list,
        description="Ranked set of member offers matching the constraints.",
    )


class CandidatePresentation(BaseModel):
    """Reducer output when multiple base products remain."""

    message: str = Field(
        ...,
        description="Question or comparison summary to present multiple options to the user.",
    )
    highlighted_keys: list[str] = Field(
        default_factory=list,
        description="List of base_random_key values mentioned explicitly in the message.",
    )


class ResolutionSummary(BaseModel):
    """Finaliser output for packaging the member recommendation."""

    message: str = Field(..., description="Final assistant message summarising the choice.")
    member_random_key: str = Field(..., description="The selected member key.")


__all__ = [
    "CandidatePresentation",
    "CategoryFeatureStatistic",
    "ClarificationPlan",
    "ConstraintExtraction",
    "FeatureConstraintModel",
    "FilteredProduct",
    "MemberFilterResponse",
    "MemberOffer",
    "ProductFilterResponse",
    "ResolutionSummary",
]

