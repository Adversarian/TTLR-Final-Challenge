"""Conversation contracts and turn state for the multi-turn assistant."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field


class StopReason(str, Enum):
    """Reasons that conclude a multi-turn interaction."""

    FOUND_UNIQUE_MEMBER = "found_unique_member"
    MAX_TURNS_REACHED = "max_turns_reached"
    USER_CANCELLED = "user_cancelled"
    RELAXATION_FAILED = "relaxation_failed"
    ROUTER_FALLBACK = "router_fallback"


class CandidatePreview(BaseModel):
    """Lightweight preview of a candidate member."""

    member_random_key: str = Field(..., description="Unique identifier for the member offer.")
    base_random_key: str = Field(..., description="Identifier of the underlying base product.")
    product_name: str = Field(..., description="Persian display name for the base product.")
    brand_name: str | None = Field(None, description="Brand label when available.")
    shop_name: str | None = Field(None, description="Name of the shop providing the offer.")
    city_name: str | None = Field(None, description="City in which the shop operates.")
    price: int | None = Field(None, description="Offer price in Tomans.")
    shop_score: float | None = Field(None, description="Seller score between 0 and 5.")


class MemberDelta(BaseModel):
    """Incremental update describing new constraints extracted from a user turn."""

    brand_names: set[str] = Field(default_factory=set)
    category_names: set[str] = Field(default_factory=set)
    city_names: set[str] = Field(default_factory=set)
    min_price: int | None = None
    max_price: int | None = None
    min_shop_score: float | None = None
    warranty_required: bool | None = None
    keywords: set[str] = Field(default_factory=set)
    product_attributes: dict[str, str] = Field(default_factory=dict)
    asked_fields: set[str] = Field(default_factory=set)
    excluded_fields: set[str] = Field(default_factory=set)
    summary: str | None = None

    drop_brand_names: bool = False
    drop_category_names: bool = False
    drop_city_names: bool = False
    drop_price_range: bool = False
    drop_min_shop_score: bool = False
    drop_warranty_requirement: bool = False
    drop_keywords: bool = False
    drop_product_attributes: bool = False

    def is_empty(self) -> bool:
        """Return ``True`` when the delta does not modify any field."""

        if (
            self.brand_names
            or self.category_names
            or self.city_names
            or self.keywords
            or self.product_attributes
            or self.asked_fields
            or self.excluded_fields
        ):
            return False

        if any(
            value is not None
            for value in (
                self.min_price,
                self.max_price,
                self.min_shop_score,
                self.warranty_required,
                self.summary,
            )
        ):
            return False

        return not any(
            (
                self.drop_brand_names,
                self.drop_category_names,
                self.drop_city_names,
                self.drop_price_range,
                self.drop_min_shop_score,
                self.drop_warranty_requirement,
                self.drop_keywords,
                self.drop_product_attributes,
            )
        )


class MemberDetails(BaseModel):
    """Aggregated hard and soft constraints describing the desired member."""

    brand_names: set[str] = Field(default_factory=set)
    category_names: set[str] = Field(default_factory=set)
    city_names: set[str] = Field(default_factory=set)
    min_price: int | None = None
    max_price: int | None = None
    min_shop_score: float | None = None
    warranty_required: bool | None = None
    keywords: list[str] = Field(default_factory=list)
    product_attributes: dict[str, str] = Field(default_factory=dict)
    asked_fields: set[str] = Field(default_factory=set)
    excluded_fields: set[str] = Field(default_factory=set)
    summary: str | None = None

    def apply_delta(self, delta: MemberDelta) -> "MemberDetails":
        """Merge the provided delta into a new MemberDetails instance."""

        updated = self.model_copy(deep=True)

        if delta.drop_brand_names:
            updated.brand_names.clear()
        updated.brand_names.update(_normalize_tokens(delta.brand_names))

        if delta.drop_category_names:
            updated.category_names.clear()
        updated.category_names.update(_normalize_tokens(delta.category_names))

        if delta.drop_city_names:
            updated.city_names.clear()
        updated.city_names.update(_normalize_tokens(delta.city_names))

        if delta.drop_price_range:
            updated.min_price = None
            updated.max_price = None
        if delta.min_price is not None:
            updated.min_price = delta.min_price
        if delta.max_price is not None:
            updated.max_price = delta.max_price

        if delta.drop_min_shop_score:
            updated.min_shop_score = None
        if delta.min_shop_score is not None:
            updated.min_shop_score = delta.min_shop_score

        if delta.drop_warranty_requirement:
            updated.warranty_required = None
        if delta.warranty_required is not None:
            updated.warranty_required = delta.warranty_required

        if delta.drop_keywords:
            updated.keywords.clear()
        if delta.keywords:
            merged_keywords = list(dict.fromkeys(updated.keywords))
            for token in _normalize_tokens(delta.keywords):
                if token not in merged_keywords:
                    merged_keywords.append(token)
            updated.keywords = merged_keywords

        if delta.drop_product_attributes:
            updated.product_attributes.clear()
        if delta.product_attributes:
            normalized_attrs = {
                key.strip(): value.strip()
                for key, value in delta.product_attributes.items()
                if key.strip() and value.strip()
            }
            updated.product_attributes.update(normalized_attrs)

        if delta.asked_fields:
            updated.asked_fields.update(delta.asked_fields)
        if delta.excluded_fields:
            updated.excluded_fields.update(delta.excluded_fields)

        if delta.summary:
            updated.summary = delta.summary

        return updated

    def extend_keywords(self, tokens: Iterable[str]) -> None:
        """Extend keywords in-place while preserving insertion order."""

        if not tokens:
            return
        merged_keywords = list(dict.fromkeys(self.keywords))
        for token in tokens:
            normalized = token.strip()
            if normalized and normalized not in merged_keywords:
                merged_keywords.append(normalized)
        self.keywords = merged_keywords


class TurnState(BaseModel):
    """Represents the compact state shared across turns."""

    turn_index: int = Field(1, ge=1, description="Current turn number (1-indexed).")
    details: MemberDetails = Field(default_factory=MemberDetails)
    asked_questions: set[str] = Field(
        default_factory=set,
        description="Identifiers of clarification prompts already presented.",
    )
    candidate_count: int | None = Field(
        None, ge=0, description="Number of candidates that match the active filters."
    )
    candidates: list[CandidatePreview] = Field(
        default_factory=list, description="Preview of the top ranked candidates."
    )
    awaiting_selection: bool = Field(
        False,
        description="Whether the assistant expects the user to pick from presented options.",
    )
    stop_reason: StopReason | None = Field(
        None, description="Reason for finishing the flow when applicable."
    )
    summary: str | None = Field(
        None,
        description=(
            "Compact textual summary of the dialogue, suitable for hydrating the"
            " next turn without replaying full transcripts."
        ),
    )

    def apply_delta(self, delta: MemberDelta) -> None:
        """Apply the provided delta to the stored member details."""

        self.details = self.details.apply_delta(delta)

    def advance_turn(self) -> None:
        """Increment the turn counter."""

        self.turn_index += 1

    def reset_candidates(self) -> None:
        """Clear cached candidate previews."""

        self.candidates.clear()
        self.candidate_count = None
        self.awaiting_selection = False


def _normalize_tokens(values: Iterable[str]) -> set[str]:
    """Return lower-cased, stripped tokens for consistent storage."""

    normalized: set[str] = set()
    for value in values:
        token = value.strip()
        if token:
            normalized.add(token)
    return normalized


__all__ = [
    "CandidatePreview",
    "MemberDetails",
    "MemberDelta",
    "StopReason",
    "TurnState",
]
