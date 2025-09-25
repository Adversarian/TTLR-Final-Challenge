"""Lightweight NLU helpers for the multi-turn assistant."""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..logging import _ensure_logfire
from .contracts import MemberDelta
from .prompts import EXTRACTION_SYSTEM_PROMPT


_EASTERN_DIGITS = {
    ord("۰"): "0",
    ord("۱"): "1",
    ord("۲"): "2",
    ord("۳"): "3",
    ord("۴"): "4",
    ord("۵"): "5",
    ord("۶"): "6",
    ord("۷"): "7",
    ord("۸"): "8",
    ord("۹"): "9",
    ord("٠"): "0",
    ord("١"): "1",
    ord("٢"): "2",
    ord("٣"): "3",
    ord("٤"): "4",
    ord("٥"): "5",
    ord("٦"): "6",
    ord("٧"): "7",
    ord("٨"): "8",
    ord("٩"): "9",
}

_ARABIC_TO_PERSIAN = {ord("ي"): "ی", ord("ك"): "ک"}
_PUNCTUATION = str.maketrans({"،": ",", "؛": ",", "٫": ".", "٬": ","})
_ZWNJ = "\u200c"
_STOPWORDS = {
    "سلام",
    "من",
    "دنبال",
    "یک",
    "يك",
    "یه",
    "هستم",
    "میخوام",
    "می‌خوام",
    "می‌خوام",
    "میخواستم",
    "می‌خواستم",
    "میخاهم",
    "می‌خواهم",
    "می‌خواهم",
    "میخواهم",
    "میخواستی",
    "کنید",
    "کنم",
    "کنیم",
    "کنین",
    "می‌تونید",
    "می‌تونید",
    "میتونید",
    "میتونین",
    "تونید",
    "میشه",
    "ممنون",
    "لطفا",
    "لطفاً",
    "برای",
    "تا",
    "که",
    "و",
    "یا",
    "با",
    "را",
    "از",
    "به",
    "در",
    "اگر",
    "آیا",
    "هم",
    "همین",
    "خیلی",
    "رو",
    "این",
    "آن",
    "اون",
    "همچنین",
    "یکم",
    "یهکم",
    "نیاز",
    "دارم",
    "داشتن",
    "باشه",
    "باشد",
    "باشند",
    "می",
    "شود",
    "بتونم",
    "بتونید",
    "بتونین",
    "خواهی",
    "خواهید",
    "بتونی",
    "دوست",
    "دارم",
    "دارید",
    "دارین",
    "کمک",
    "کمکم",
    "راهنمایی",
    "راهنما",
    "سریع",
    "دوست",
    "دارید",
}
_STOPWORDS_LOWER = {word.lower() for word in _STOPWORDS}
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9آ-ی]+")
_PRICE_RANGE_PATTERN = re.compile(
    r"(?P<low>\d[\d,]*)\s*(?P<low_unit>میلیون|هزار)?\s*(?:تا|الی|الیه|-|و)\s*(?P<high>\d[\d,]*)\s*(?P<high_unit>میلیون|هزار)?\s*(?:تومان|تومن|ریال)?"
)
_PRICE_SINGLE_PATTERN = re.compile(
    r"(?:(?:قیمت|حدود|تقریباً|نزدیک)\s*)?(?P<value>\d[\d,]{3,})\s*(?P<unit>میلیون|هزار)?\s*(?:تومان|تومن|ریال)?"
)
_DONT_CARE_PATTERNS = {
    "price": [r"قیمت\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد|فرقی\s+نمی\s*کند)"],
    "brand": [r"برند\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد|فرقی\s+نمی\s*کند)"],
    "city": [r"(?:شهر|مکان)\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد|فرقی\s+نمی\s*کند)"],
    "warranty": [r"گارانتی\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد)"],
    "score": [r"امتیاز\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد)"],
    "category": [r"(?:دسته|نوع|مدل)\s*(?:اصلا\s*)?(?:مهم\s+نیست|اهمیتی\s+ندارد)"],
}
_MATERIAL_PATTERN = re.compile(r"جنس(?:ش)?\s*(?:=|:)?\s*(?P<value>[آ-یA-Za-z0-9\s]{2,})")
_COLOR_PATTERN = re.compile(r"رنگ(?:ش)?\s*(?:=|:)?\s*(?P<value>[آ-یA-Za-z0-9\s]{2,})")
_SCORE_PATTERN = re.compile(
    r"امتیاز\s*(?:بالای?|حداقل)?\s*(?P<value>\d(?:\.\d)?)?\s*(?:یا\s*بیشتر|به\s*بالا)?"
)
_HIGH_SCORE_HINTS = ["امتیاز بالا", "امتیاز بالایی", "امتیاز خوب", "فروشنده خوب"]
_SUMMARY_LIMIT = 200


@dataclass(slots=True)
class _AgentCache:
    """Container that allows tests to reset the cached extractor agent."""

    agent: Agent["LLMExtraction"] | None = None
    initialised: bool = False


_CACHE = _AgentCache()


class NLUDelta(BaseModel):
    """Wrapper around the extracted member delta."""

    delta: MemberDelta = Field(default_factory=MemberDelta)


class NLUResult(NLUDelta):
    """Structured representation of the lightweight NLU outcome."""

    original_text: str = ""
    normalized_text: str = ""
    tokens: list[str] = Field(default_factory=list)


class LLMExtraction(BaseModel):
    """Schema returned by the LLM-backed extractor."""

    brand_names: list[str] = Field(default_factory=list)
    category_names: list[str] = Field(default_factory=list)
    city_names: list[str] = Field(default_factory=list)
    min_price: int | None = None
    max_price: int | None = None
    min_shop_score: float | None = None
    warranty_required: bool | None = None
    keywords: list[str] = Field(default_factory=list)
    product_attributes: dict[str, str] = Field(default_factory=dict)
    asked_fields: list[str] = Field(default_factory=list)
    excluded_fields: list[str] = Field(default_factory=list)
    drop_brand_names: bool = False
    drop_category_names: bool = False
    drop_city_names: bool = False
    drop_price_range: bool = False
    drop_min_shop_score: bool = False
    drop_warranty_requirement: bool = False
    drop_keywords: bool = False
    drop_product_attributes: bool = False
    summary: str | None = None

    def to_delta(self, *, normalized_text: str, fallback_tokens: Sequence[str]) -> MemberDelta:
        """Convert the extraction payload into a :class:`MemberDelta`."""

        delta = MemberDelta(
            brand_names=set(self.brand_names),
            category_names=set(self.category_names),
            city_names=set(self.city_names),
            min_price=self.min_price,
            max_price=self.max_price,
            min_shop_score=self.min_shop_score,
            warranty_required=self.warranty_required,
            keywords=set(self.keywords),
            product_attributes=dict(self.product_attributes),
            asked_fields=set(self.asked_fields),
            excluded_fields=set(self.excluded_fields),
            summary=self.summary,
            drop_brand_names=self.drop_brand_names,
            drop_category_names=self.drop_category_names,
            drop_city_names=self.drop_city_names,
            drop_price_range=self.drop_price_range,
            drop_min_shop_score=self.drop_min_shop_score,
            drop_warranty_requirement=self.drop_warranty_requirement,
            drop_keywords=self.drop_keywords,
            drop_product_attributes=self.drop_product_attributes,
        )

        if not delta.summary and normalized_text:
            delta.summary = normalized_text[:_SUMMARY_LIMIT]

        if not delta.keywords and fallback_tokens:
            delta.keywords.update(fallback_tokens)

        return delta


def get_nlu_agent() -> Agent[LLMExtraction] | None:
    """Return the cached LLM extractor agent when credentials are available."""

    global _CACHE

    if _CACHE.initialised:
        return _CACHE.agent

    _ensure_logfire()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key and not base_url:
        _CACHE = _AgentCache(agent=None, initialised=True)
        return None

    model_name = (
        os.getenv("OPENAI_MULTITURN_NLU_MODEL")
        or os.getenv("OPENAI_ROUTER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1"
    )

    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        settings=ModelSettings(temperature=0.0, parallel_tool_calls=False),
    )

    agent = Agent(
        model=model,
        output_type=LLMExtraction,
        instructions=EXTRACTION_SYSTEM_PROMPT,
        instrument=InstrumentationSettings(),
        name="multiturn-nlu-extractor",
    )

    _CACHE = _AgentCache(agent=agent, initialised=True)
    return agent


def reset_nlu_agent_cache() -> None:
    """Reset the cached LLM agent (used in tests)."""

    global _CACHE
    _CACHE = _AgentCache()


def normalize_text(text: str) -> str:
    """Normalize Persian digits and spacing for downstream parsing."""

    if not text:
        return ""
    normalized = text.translate(_EASTERN_DIGITS).translate(_ARABIC_TO_PERSIAN)
    normalized = normalized.translate(_PUNCTUATION)
    normalized = normalized.replace(_ZWNJ, " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def parse_user_message(text: str) -> NLUResult:
    """Parse a Persian user utterance into structured constraints."""

    normalized = normalize_text(text)
    tokens = _extract_keywords(normalized)

    agent = get_nlu_agent()

    if agent is not None:
        try:
            llm_result = agent.run_sync(
                normalized,
                original_text=text,
            )
            extraction = llm_result.output
            delta = extraction.to_delta(normalized_text=normalized, fallback_tokens=tokens)
        except Exception:  # pragma: no cover - defensive fallback
            delta = _rule_based_delta(normalized, tokens)
    else:
        delta = _rule_based_delta(normalized, tokens)

    if not delta.keywords and tokens:
        delta.keywords.update(tokens)

    result_tokens = list(delta.keywords) if delta.keywords else list(tokens)

    return NLUResult(
        original_text=text,
        normalized_text=normalized,
        tokens=result_tokens,
        delta=delta,
    )


def _rule_based_delta(normalized: str, tokens: Sequence[str]) -> MemberDelta:
    """Fallback rule-based extraction used when the LLM is unavailable."""

    delta = MemberDelta()

    price_range = _extract_price_range(normalized)
    if price_range:
        min_price, max_price = price_range
        if min_price is not None:
            delta.min_price = min_price
        if max_price is not None:
            delta.max_price = max_price
        delta.asked_fields.add("price")

    warranty = _extract_warranty(normalized)
    if warranty is not None:
        delta.warranty_required = warranty
        delta.asked_fields.add("warranty")

    score = _extract_min_score(normalized)
    if score is not None:
        delta.min_shop_score = score
        delta.asked_fields.add("score")

    attributes = _extract_product_attributes(normalized)
    if attributes:
        delta.product_attributes.update(attributes)

    dont_care_slots = _extract_dont_care(normalized)
    if dont_care_slots:
        delta.excluded_fields.update(dont_care_slots)

    if tokens:
        delta.keywords.update(tokens)

    if normalized:
        delta.summary = normalized[:_SUMMARY_LIMIT]

    return delta


def _extract_keywords(text: str) -> list[str]:
    """Return candidate keywords suitable for fuzzy catalogue search."""

    seen = set()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token.isdigit():
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        if lowered in _STOPWORDS_LOWER:
            continue
        seen.add(lowered)
        tokens.append(token)
    return tokens


def _extract_price_range(text: str) -> tuple[int | None, int | None] | None:
    """Extract a price range expressed in Tomans from the text."""

    match = _PRICE_RANGE_PATTERN.search(text)
    if match:
        low = _resolve_number(match.group("low"), match.group("low_unit"))
        high = _resolve_number(match.group("high"), match.group("high_unit"))
        if low is not None and high is not None:
            if low > high:
                low, high = high, low
            return low, high
        if low is not None or high is not None:
            return low, high

    single_candidates: list[int] = []
    for match in _PRICE_SINGLE_PATTERN.finditer(text):
        value = _resolve_number(match.group("value"), match.group("unit"))
        if value is None:
            continue
        span_start, span_end = match.span()
        context_start = max(0, span_start - 12)
        context_end = min(len(text), span_end + 12)
        context = text[context_start:context_end]
        if re.search(r"قیمت|تومان|تومن", context):
            single_candidates.append(value)

    if single_candidates:
        value = single_candidates[0]
        return value, value

    return None


def _resolve_number(value: str | None, unit: str | None) -> int | None:
    if not value:
        return None
    digits = value.replace(",", "")
    if not digits:
        return None
    number = int(digits)
    if unit:
        if unit.startswith("میلیون"):
            number *= 1_000_000
        elif unit.startswith("هزار"):
            number *= 1_000
    return number


def _extract_warranty(text: str) -> bool | None:
    if not text:
        return None
    if re.search(r"بدون\s+گارانتی|گارانتی\s*(?:ندارد|نمی\s*خوام|نمی\s*خواهم)", text):
        return False
    if re.search(r"گارانتی\s*(?:دار(?:د|ه)|داشته|می\s*خوام|می\s*خواهم|باشه)", text):
        return True
    return None


def _extract_min_score(text: str) -> float | None:
    if not text:
        return None
    match = _SCORE_PATTERN.search(text)
    if match and match.group("value"):
        value = float(match.group("value"))
        if math.isnan(value) or math.isinf(value):
            return None
        if value > 5:
            value = 5.0
        return value
    for hint in _HIGH_SCORE_HINTS:
        if hint in text:
            return 4.0
    return None


def _extract_product_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}

    material_match = _MATERIAL_PATTERN.search(text)
    if material_match:
        attributes["material"] = _clean_attribute_value(material_match.group("value"))

    color_match = _COLOR_PATTERN.search(text)
    if color_match:
        attributes["color"] = _clean_attribute_value(color_match.group("value"))

    if "ارسال رایگان" in text or "ارسال گل رایگان" in text:
        attributes["shipping"] = "free"

    return {key: value for key, value in attributes.items() if value}


def _clean_attribute_value(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"(?:\s+و\s+|\s+باش(?:ه|د)|\s+هست|,|\.|،)", value)[0]
    value = re.sub(r"\b(?:داشته|دارای)\b$", "", value).strip()
    return value


def _extract_dont_care(text: str) -> set[str]:
    dont_care: set[str] = set()
    for slot, patterns in _DONT_CARE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                dont_care.add(slot)
                break
    return dont_care


__all__ = [
    "LLMExtraction",
    "NLUDelta",
    "NLUResult",
    "get_nlu_agent",
    "normalize_text",
    "parse_user_message",
    "reset_nlu_agent_cache",
]
