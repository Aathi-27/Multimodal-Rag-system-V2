"""
Text Normalization - Cleaning pipeline applied identically at ingestion and query.

Rules (from PRD):
- Unicode normalization (NFC)
- Standardize whitespace (tabs → spaces, collapse multiples)
- Normalize quotes and apostrophes
- MUST be identical for ingestion and query processing
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Quote normalization map
QUOTE_MAP = str.maketrans({
    "\u2018": "'",   # Left single quote
    "\u2019": "'",   # Right single quote
    "\u201C": '"',   # Left double quote
    "\u201D": '"',   # Right double quote
    "\u2013": "-",   # En dash
    "\u2014": "-",   # Em dash
    "\u2026": "...", # Ellipsis
    "\u00A0": " ",   # Non-breaking space
})

# Regex patterns (compiled once)
RE_MULTIPLE_SPACES = re.compile(r" {2,}")
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
RE_TABS = re.compile(r"\t")


def normalize_text(text: str) -> str:
    """
    Apply the authoritative text normalization pipeline.

    This function MUST produce identical output whether called
    during ingestion or during query processing.

    Pipeline:
    1. Unicode NFC normalization
    2. Quote and dash normalization
    3. Tab → space
    4. Collapse multiple spaces
    5. Collapse excessive newlines (3+ → 2)
    6. Strip leading/trailing whitespace

    Args:
        text: Raw input text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize quotes, dashes, special chars
    text = text.translate(QUOTE_MAP)

    # 3. Tabs to spaces
    text = RE_TABS.sub(" ", text)

    # 4. Collapse multiple spaces (preserve newlines)
    text = RE_MULTIPLE_SPACES.sub(" ", text)

    # 5. Collapse excessive newlines
    text = RE_MULTIPLE_NEWLINES.sub("\n\n", text)

    # 6. Strip
    text = text.strip()

    return text


def normalize_query(query: str) -> str:
    """
    Normalize a user query.
    Uses the SAME pipeline as ingestion to ensure consistency.
    """
    return normalize_text(query)
