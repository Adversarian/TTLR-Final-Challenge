"""Schema definitions for the conversation router."""

from __future__ import annotations

from typing import Any, Mapping, Literal

from pydantic import BaseModel, Field, model_validator


class RouterDecision(BaseModel):
    """Binary routing choice indicating whether multi-turn handling is needed."""

    route: Literal["single_turn", "multi_turn"] = Field(
        ..., description="Whether the query can be handled in one turn or needs dialogue.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_label(cls, value: Any) -> Any:
        """Allow bare label responses from the model to populate the schema."""

        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return {"route": stripped}
        if isinstance(value, Mapping):
            return value
        return value


__all__ = ["RouterDecision"]
