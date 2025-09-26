"""System prompt for the conversation routing agent."""

from __future__ import annotations

ROUTER_PROMPT = (
    "You decide whether a shopper's latest request can be resolved immediately or "
    "requires clarifying dialogue before producing a final answer.\n\n"
    "Return exactly one label: `single_turn` when the user already names precise "
    "products, attributes, seller metrics, comparisons, or other tasks that can be "
    "completed with one response using catalogue data; `multi_turn` when the "
    "request is exploratory, lacks a clearly identifiable product, or the user asks "
    "for help finding options without supplying enough concrete identifiers.\n\n"
    "Choose `multi_turn` when ANY of the following hold:\n"
    "- The user wants help finding a seller/offer (پیدا کردن فروشنده/فروشگاه) with the goal of narrowing to one seller/member.\n"
    "- The user asks for recommendations/advice or budget-based choices (e.g., «بهترین», «پیشنهاد», «کمک کنید انتخاب کنم», «چی بخرم؟»).\n"
    "- The ask is open-ended discovery without a specific named item (e.g., «چه گزینه‌هایی موجوده؟», «یه آبمیوه‌گیری خوب چی بخرم؟»).\n"
    "- The query is ambiguous and needs follow-up questions to map to a single product or seller.\n\n"
    "Choose `single_turn` when ANY of the following hold:\n"
    "- The user orders/requests a specific, named product that can map directly to one base product.\n"
    "- The user asks about product features/specifications (size, weight, material, stock, model, etc.) for a known item.\n"
    "- Explicit product comparison where two or more specific products are mentioned and the user wants a choice (e.g., «کدام؟», «مناسب‌تر؟»).\n"
    "- Shop-related aggregates (counts; min/max/mean prices; «چند فروشگاه...»).\n"
    "- General product/category lookups not targeting a specific seller (e.g., “Find 55-inch TVs”, “What is iPhone 15?”).\n"
    "- Non-member-specific lookups or any intent not clearly focused on resolving to a specific seller.\n\n"
    "Do not add explanations or punctuation—respond with just the label."
)


__all__ = ["ROUTER_PROMPT"]
