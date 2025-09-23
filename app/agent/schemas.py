"""Pydantic models used by the shopping assistant agent."""

from __future__ import annotations

from decimal import Decimal
from typing import List

from pydantic import BaseModel, Field


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


class ProductFeature(BaseModel):
    """Single flattened feature/value pair for a product."""

    name: str = Field(..., description="Feature label as stored in the catalogue.")
    value: str = Field(..., description="Human readable value for the feature.")


class FeatureLookupResult(BaseModel):
    """Complete catalogue feature map for a base product."""

    base_random_key: str = Field(..., description="Target product random key.")
    features: List[ProductFeature] = Field(
        default_factory=list,
        description="All feature/value pairs extracted from the catalogue record.",
    )
    available_features: List[str] = Field(
        default_factory=list,
        description=(
            "Convenience list of feature names to help pick the relevant attribute."
        ),
    )


class CitySellerStatistics(BaseModel):
    """Aggregated seller metrics for a specific city."""

    city_id: int | None = Field(None, description="Identifier of the city.")
    city_name: str | None = Field(None, description="Display name of the city.")
    offer_count: int = Field(..., ge=0, description="Total offers from this city.")
    distinct_shops: int = Field(
        ..., ge=0, description="Number of unique shops contributing offers."
    )
    shops_with_warranty: int = Field(
        ..., ge=0, description="Offers backed by Torob warranty."
    )
    shops_without_warranty: int = Field(
        ..., ge=0, description="Offers without Torob warranty."
    )
    min_price: int | None = Field(
        None, description="Cheapest price observed in this city."
    )
    max_price: int | None = Field(
        None, description="Most expensive price observed in this city."
    )
    average_price: float | None = Field(
        None, description="Average offer price for this city."
    )
    min_score: float | None = Field(
        None, description="Lowest shop score observed in this city."
    )
    max_score: float | None = Field(
        None, description="Highest shop score observed in this city."
    )
    average_score: float | None = Field(
        None, description="Average shop score for this city."
    )


class SellerStatistics(BaseModel):
    """Aggregated marketplace data for a base product."""

    base_random_key: str = Field(..., description="Target base product key.")
    city: str | None = Field(
        None,
        description="Optional Persian city name used to focus the aggregation.",
    )
    total_offers: int = Field(..., ge=0, description="Total number of offers.")
    distinct_shops: int = Field(
        ..., ge=0, description="Number of unique shops listing the product."
    )
    shops_with_warranty: int = Field(
        ..., ge=0, description="Offers sold with Torob warranty."
    )
    shops_without_warranty: int = Field(
        ..., ge=0, description="Offers sold without Torob warranty."
    )
    min_price: int | None = Field(
        None, description="Cheapest price offered across all shops."
    )
    max_price: int | None = Field(
        None, description="Most expensive price offered across all shops."
    )
    average_price: float | None = Field(
        None, description="Average price across all offers."
    )
    min_score: float | None = Field(
        None, description="Lowest shop score for the product offers."
    )
    max_score: float | None = Field(
        None, description="Highest shop score for the product offers."
    )
    average_score: float | None = Field(
        None, description="Average shop score across offers."
    )
    num_cities_with_offers: int = Field(
        ..., ge=0, description="Number of cities that list the product."
    )
    city_stats: List[CitySellerStatistics] = Field(
        default_factory=list,
        description="Optional per-city breakdowns to support city-specific answers.",
    )


class SellerOffer(BaseModel):
    """Represents an individual seller listing for a base product."""

    member_random_key: str = Field(
        ..., description="Identifier of the seller's specific listing."
    )
    shop_id: int = Field(..., description="Identifier of the shop offering the product.")
    price: int = Field(..., ge=0, description="Offer price in Tomans.")
    shop_score: float | None = Field(
        None, description="Shop score as recorded on the platform."
    )
    has_warranty: bool = Field(
        ..., description="Whether the listing advertises Torob warranty."
    )
    city_id: int | None = Field(
        None, description="Identifier of the city the shop operates in."
    )
    city_name: str | None = Field(
        None, description="Display name of the city the shop operates in."
    )


class SellerOfferList(BaseModel):
    """Collection of seller offers for a base product."""

    base_random_key: str = Field(..., description="Target base product key.")
    offers: List[SellerOffer] = Field(
        default_factory=list,
        description="Short list of seller offers ordered by the requested ranking.",
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
    numeric_answer: Decimal | None = Field(
        None,
        description=(
            "When responding with a numeric seller statistic, populate this field "
            "with the exact value so the API layer can enforce digit-only replies."
        ),
    )

    def clipped(self) -> "AgentReply":
        """Return a copy trimmed to the API list length limits."""

        return AgentReply(
            message=self.message,
            base_random_keys=self.base_random_keys[:10],
            member_random_keys=self.member_random_keys[:10],
            numeric_answer=self.numeric_answer,
        )


__all__ = [
    "ProductMatch",
    "ProductSearchResult",
    "ProductFeature",
    "FeatureLookupResult",
    "CitySellerStatistics",
    "SellerStatistics",
    "SellerOffer",
    "SellerOfferList",
    "AgentReply",
]
