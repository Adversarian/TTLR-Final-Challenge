"""Tooling primitives consumed by the Torob shopping assistant."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool

from .models import ProductLookupArgs
from .retrieval import gather_product_contexts


def _coerce_lookup_args(
    payload: object | None,
    *,
    product_name: Optional[str] = None,
    base_random_key: Optional[str] = None,
    member_random_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> ProductLookupArgs:
    if isinstance(payload, ProductLookupArgs):
        lookup_args = payload
    else:
        data: Dict[str, Any] = {}
        if isinstance(payload, dict):
            data.update(payload)
        elif payload is not None:
            raise TypeError(
                "lookup_products payload must be a dict or ProductLookupArgs; "
                f"received {type(payload)!r}"
            )

        merged_kwargs: Dict[str, Any] = {
            "product_name": product_name,
            "base_random_key": base_random_key,
            "member_random_key": member_random_key,
        }
        for key, value in merged_kwargs.items():
            if key not in data:
                data[key] = value
        if "limit" not in data and limit is not None:
            data["limit"] = limit

        lookup_args = ProductLookupArgs.model_validate(data)
    return lookup_args


@lru_cache(maxsize=1)
def get_lookup_tool() -> FunctionTool:
    def lookup_products(
        payload: object | None = None,
        *,
        product_name: Optional[str] = None,
        base_random_key: Optional[str] = None,
        member_random_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Hybrid search over the product catalog with feature and seller context."""

        lookup_args = _coerce_lookup_args(
            payload,
            product_name=product_name,
            base_random_key=base_random_key,
            member_random_key=member_random_key,
            limit=limit,
        )

        contexts = gather_product_contexts(lookup_args)
        return [context.model_dump(mode="json") for context in contexts]

    return FunctionTool.from_defaults(
        fn=lookup_products,
        name="lookup_products",
        description=(
            "Retrieve Torob products using hybrid semantic + fuzzy search. "
            "Always call this once per turn to obtain product features and seller stats."
        ),
    )


__all__ = ["get_lookup_tool"]
