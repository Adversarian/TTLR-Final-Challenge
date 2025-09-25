"""Schema definitions for the turn-taking router."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Classification emitted by the router to guide downstream handling."""

    mode: Literal["single_turn", "multi_turn"] = Field(
        ..., description="Whether the request can be answered immediately or needs follow-up.",
    )


__all__ = ["RoutingDecision"]
