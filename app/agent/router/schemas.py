"""Schema definitions for the conversation router."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Binary routing choice indicating whether multi-turn handling is needed."""

    route: Literal["single_turn", "multi_turn"] = Field(
        ..., description="Whether the query can be handled in one turn or needs dialogue.",
    )


__all__ = ["RouterDecision"]
