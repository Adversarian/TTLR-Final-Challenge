"""Schema definitions for the vision router."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, model_validator


class VisionRouteDecision(BaseModel):
    """Routing choice for image-containing requests."""

    route: Literal["explanation", "similarity"] = Field(
        ...,
        description="Whether the image query needs a description or similar products.",
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


__all__ = ["VisionRouteDecision"]
