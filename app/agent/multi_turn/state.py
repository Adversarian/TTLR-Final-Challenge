"""Conversation state tracking utilities for the scenario 4 workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from pydantic_ai.messages import ModelMessage

from .schemas import ConstraintExtraction, FeatureConstraintModel, FilteredProduct, MemberOffer


def _normalise_text(value: str) -> str:
    """Return a simplified representation for deduplication purposes."""

    replacements = {
        "\u064a": "ی",
        "\u0643": "ک",
        "\u06cc": "ی",
        "\u06a9": "ک",
    }
    lowered = value.strip().lower()
    for src, dest in replacements.items():
        lowered = lowered.replace(src, dest)
    return lowered


@dataclass(slots=True)
class FeatureRequirement:
    """Represents a constraint the customer expects to be honoured."""

    name: str
    value: str
    match: str

    def key(self) -> Tuple[str, str, str]:
        """Return a stable deduplication key."""

        return (_normalise_text(self.name), _normalise_text(self.value), self.match)


class _FeatureBucket:
    """Utility container avoiding duplicate feature requirements."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str, str], FeatureRequirement] = {}

    def add(self, feature: FeatureRequirement) -> None:
        """Insert a new requirement unless an identical one already exists."""

        key = feature.key()
        if key not in self._items:
            self._items[key] = feature

    def extend(self, features: Iterable[FeatureRequirement]) -> None:
        """Bulk insert multiple requirements."""

        for feature in features:
            self.add(feature)

    def values(self) -> List[FeatureRequirement]:
        """Return the stored requirements preserving insertion order."""

        return list(self._items.values())

    def clear(self) -> None:
        """Remove all recorded requirements."""

        self._items.clear()


@dataclass
class ConstraintState:
    """Aggregated representation of all customer preferences so far."""

    category_hint: str | None = None
    brand_preferences: set[str] = field(default_factory=set)
    price_min: int | None = None
    price_max: int | None = None
    require_warranty: bool | None = None
    min_shop_score: float | None = None
    city_preferences: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    summaries: List[str] = field(default_factory=list)
    required_features: _FeatureBucket = field(default_factory=_FeatureBucket)
    optional_features: _FeatureBucket = field(default_factory=_FeatureBucket)
    excluded_features: _FeatureBucket = field(default_factory=_FeatureBucket)

    def apply_update(self, extraction: ConstraintExtraction) -> None:
        """Merge a new extraction result into the running state."""

        summary = extraction.summary.strip()
        if summary:
            self.summaries.append(summary)

        if extraction.category_hint:
            category = extraction.category_hint.strip()
            if category:
                # Prefer the first stated category to avoid oscillations.
                if not self.category_hint:
                    self.category_hint = category

        for brand in extraction.brand_preferences:
            normalised = _normalise_text(brand)
            if normalised:
                self.brand_preferences.add(brand.strip())

        if extraction.price_min is not None:
            candidate = int(extraction.price_min)
            if self.price_min is None or candidate > self.price_min:
                self.price_min = candidate

        if extraction.price_max is not None:
            candidate = int(extraction.price_max)
            if self.price_max is None or candidate < self.price_max:
                self.price_max = candidate

        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            # Swap the values when user inverted the range.
            self.price_min, self.price_max = self.price_max, self.price_min

        if extraction.require_warranty is True:
            self.require_warranty = True
        elif extraction.require_warranty is False and self.require_warranty is None:
            self.require_warranty = False

        if extraction.min_shop_score is not None:
            score = float(extraction.min_shop_score)
            if self.min_shop_score is None or score > self.min_shop_score:
                self.min_shop_score = score

        for city in extraction.city_preferences:
            stripped = city.strip()
            if stripped:
                self.city_preferences.add(stripped)

        for keyword in extraction.keywords:
            stripped = keyword.strip()
            if stripped:
                self.keywords.add(stripped)

        self.required_features.extend(
            _convert_features(extraction.required_features)
        )
        self.optional_features.extend(
            _convert_features(extraction.optional_features)
        )
        self.excluded_features.extend(
            _convert_features(extraction.excluded_features)
        )

    def snapshot(self) -> dict[str, object]:
        """Return a serialisable snapshot for prompt construction."""

        return {
            "category_hint": self.category_hint,
            "brand_preferences": sorted(self.brand_preferences),
            "price_min": self.price_min,
            "price_max": self.price_max,
            "require_warranty": self.require_warranty,
            "min_shop_score": self.min_shop_score,
            "city_preferences": sorted(self.city_preferences),
            "keywords": sorted(self.keywords),
            "required_features": [feature.__dict__ for feature in self.required_features.values()],
            "optional_features": [feature.__dict__ for feature in self.optional_features.values()],
            "excluded_features": [feature.__dict__ for feature in self.excluded_features.values()],
            "summaries": list(self.summaries[-3:]),
        }


def _convert_features(features: Iterable[FeatureConstraintModel]) -> List[FeatureRequirement]:
    """Convert Pydantic feature constraints into internal dataclasses."""

    converted: List[FeatureRequirement] = []
    for feature in features:
        name = feature.name.strip()
        value = feature.value.strip()
        match = feature.match.strip()
        if not name or not value:
            continue
        converted.append(FeatureRequirement(name=name, value=value, match=match))
    return converted


@dataclass
class Scenario4ConversationState:
    """Complete runtime context for a scenario 4 dialogue."""

    chat_id: str
    max_turns: int = 5
    turn_count: int = 0
    constraints: ConstraintState = field(default_factory=ConstraintState)
    asked_questions: List[str] = field(default_factory=list)
    candidate_products: List[FilteredProduct] = field(default_factory=list)
    candidate_offers: List[MemberOffer] = field(default_factory=list)
    locked_base_key: str | None = None
    finalized_member_key: str | None = None
    latest_user_message: str | None = None
    agent_histories: Dict[str, List[ModelMessage]] = field(default_factory=dict)
    completed: bool = False

    def next_turn(self) -> None:
        """Increment the assistant turn counter."""

        self.turn_count += 1

    def remaining_turns(self) -> int:
        """Return how many assistant responses remain before hitting the cap."""

        return max(self.max_turns - self.turn_count, 0)

    def reset(self) -> None:
        """Clear any accumulated candidates when starting a new branch."""

        self.candidate_products.clear()
        self.candidate_offers.clear()
        self.locked_base_key = None
        self.finalized_member_key = None
        self.completed = False


__all__ = [
    "ConstraintState",
    "FeatureRequirement",
    "Scenario4ConversationState",
]

