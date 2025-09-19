"""Lightweight text normalization helpers."""
from __future__ import annotations

import re
import unicodedata

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans({
    **{ord(p): ASCII_DIGITS[i] for i, p in enumerate(PERSIAN_DIGITS)},
    **{ord(a): ASCII_DIGITS[i] for i, a in enumerate(ARABIC_DIGITS)},
})

LETTER_TRANSLATION = str.maketrans({
    ord("ي"): "ی",
    ord("ك"): "ک",
    ord("ۀ"): "ه",
    ord("ة"): "ه",
})

WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize Persian/Arabic text and digits for FTS use."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(LETTER_TRANSLATION)
    normalized = normalized.translate(DIGIT_TRANSLATION)
    normalized = normalized.replace("\u200c", "")  # remove zero-width joiner
    normalized = normalized.lower()
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = normalized.strip()
    return normalized
