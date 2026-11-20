"""Arabic-aware text preprocessing (spec section 5).

Two levels are provided:

* ``normalize_arabic`` – light normalization safe for display and indexing:
  removes tatweel and diacritics and unifies alef forms. Medical meaning is
  never altered.
* ``normalize_for_match`` – stronger normalization used only for matching
  and duplicate detection (letter unification, punctuation removal,
  whitespace collapsing, Latin lower-casing). Original text is always
  preserved for display.
"""
from __future__ import annotations

import re
import unicodedata

_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[ً-ٰٟ]")
_ALEF_VARIANTS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
})
_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)
_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")


def normalize_arabic(text: str) -> str:
    """Light, meaning-preserving Arabic normalization."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace(_TATWEEL, "")
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normalization used for search keys and deduplication."""
    text = normalize_arabic(text).lower()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def contains_arabic(text: str) -> bool:
    return bool(text) and bool(_ARABIC_CHARS.search(str(text)))


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0600" <= c <= "\u06FF") / len(letters)
