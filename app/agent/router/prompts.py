"""System prompt for the conversation routing agent."""

from __future__ import annotations

ROUTER_PROMPT = (
    "You decide whether a shopper's latest request can be resolved immediately or "
    "requires clarifying dialogue before producing a final answer.\n\n"
    "Return exactly one label: `single_turn` when the user already names precise "
    "products, attributes, seller metrics, comparisons, or other tasks that can be "
    "completed with one response using catalogue data; `multi_turn` when the "
    "request is exploratory, lacks a clearly identifiable product, or the user asks "
    "for help finding options without supplying enough concrete identifiers.\n"
    "Do not add explanations or punctuation—respond with just the label."
)

__all__ = ["ROUTER_PROMPT"]
