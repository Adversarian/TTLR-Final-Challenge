"""Utilities for working with base product feature dictionaries."""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Tuple

from .preprocess import normalize_query

ATTRIBUTE_SYNONYMS = {
    "width": ["عرض", "پهنا", "width"],
    "length": ["طول", "درازا", "length"],
    "height": ["ارتفاع", "بلندی", "height"],
    "weight": ["وزن", "سنگینی", "weight"],
    "wattage": ["توان", "وات", "watt", "w", "توان مصرفی"],
    "volume": ["حجم", "گنجایش", "liter", "لیتر", "L", "ml"],
}

UNIT_CONVERSIONS = {
    ("mm", "cm"): 0.1,
    ("cm", "mm"): 10.0,
    ("cm", "m"): 0.01,
    ("m", "cm"): 100.0,
    ("g", "kg"): 0.001,
    ("kg", "g"): 1000.0,
    ("ml", "l"): 0.001,
    ("l", "ml"): 1000.0,
    ("w", "kw"): 0.001,
    ("kw", "w"): 1000.0,
}

STANDARD_UNITS = {
    "mm": "mm",
    "millimeter": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "m": "m",
    "meter": "m",
    "g": "g",
    "gram": "g",
    "kg": "kg",
    "kilogram": "kg",
    "w": "w",
    "watt": "w",
    "kw": "kw",
    "kilowatt": "kw",
    "l": "l",
    "liter": "l",
    "litre": "l",
    "ml": "ml",
    "milliliter": "ml",
    "millilitre": "ml",
}


def find_attribute_key(attribute_phrase: str, keys: Iterable[str]) -> Optional[str]:
    """Match a phrase to a canonical attribute key."""

    normalized = normalize_query(attribute_phrase)
    for canonical, synonyms in ATTRIBUTE_SYNONYMS.items():
        if normalized in (normalize_query(s) for s in synonyms):
            for key in keys:
                if normalize_query(key) in (normalize_query(s) for s in synonyms):
                    return key
            return canonical
    # fallback exact/contains
    for key in keys:
        if normalized == normalize_query(key):
            return key
    for key in keys:
        if normalize_query(key).find(normalized) != -1:
            return key
    return None


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Map arbitrary unit tokens to a standard representation."""

    if unit is None:
        return None
    normalized = normalize_query(unit)
    return STANDARD_UNITS.get(normalized, unit)


def convert_value(value: float, source_unit: Optional[str], target_unit: Optional[str]) -> Tuple[float, Optional[str]]:
    """Convert a numeric value to the target unit when supported."""

    if source_unit is None or target_unit is None or source_unit == target_unit:
        return value, target_unit or source_unit
    factor = UNIT_CONVERSIONS.get((source_unit, target_unit))
    if factor is not None:
        return value * factor, target_unit
    return value, source_unit


def extract_value(feature_value: Any) -> Tuple[Optional[float], Optional[str], str]:
    """Extract numeric value, unit, and text description from a feature entry."""

    if isinstance(feature_value, (int, float)):
        return float(feature_value), None, str(feature_value)
    if isinstance(feature_value, str):
        return _parse_numeric_from_text(feature_value)
    if isinstance(feature_value, dict):
        raw_value = feature_value.get("value")
        unit = feature_value.get("unit")
        text = feature_value.get("text") or str(raw_value)
        if isinstance(raw_value, (int, float)):
            return float(raw_value), normalize_unit(unit), text
    return None, None, str(feature_value)


NUMERIC_RE = re.compile(r"(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>[\wآ-ی]+)?", re.UNICODE)


def _parse_numeric_from_text(text: str) -> Tuple[Optional[float], Optional[str], str]:
    match = NUMERIC_RE.search(text)
    if not match:
        return None, None, text
    value = match.group("value").replace(",", ".")
    unit = match.group("unit")
    try:
        number = float(value)
    except ValueError:
        return None, normalize_unit(unit), text
    return number, normalize_unit(unit), text
