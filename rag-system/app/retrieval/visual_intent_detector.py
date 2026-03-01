"""
Visual Intent Detector — Keyword-based gating for CLIP image branch.

Before running the CLIP search branch (extra embedding + extra Qdrant query),
detect whether the query contains visual-intent cues:
  "chart", "diagram", "screenshot", "image", "graph", "figure", ...

If no visual cues are detected → skip the image branch entirely.
This saves latency by avoiding dual-search on text-only queries.

Implementation:
  - Simple keyword matching (fast, deterministic, no ML overhead)
  - Keywords are configurable via ``clip.visual_keywords`` in config.yaml
  - Returns confidence score (0.0 = no visual intent, 1.0 = strong)
"""

from __future__ import annotations

import re
import logging

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Pre-compiled pattern (set at module load, updated per-call if needed)
_pattern_cache: dict[str, re.Pattern] = {}


def detect_visual_intent(query: str) -> dict:
    """
    Detect whether a query has visual intent (implies image search).

    Args:
        query: The user's query text.

    Returns:
        Dict with:
            has_visual_intent (bool) — Should image branch be executed?
            confidence (float) — 0.0 to 1.0
            matched_keywords (list[str]) — Which keywords matched
    """
    settings = get_settings()
    keywords = settings.clip.visual_keywords

    if not keywords:
        return {"has_visual_intent": False, "confidence": 0.0, "matched_keywords": []}

    # Build or retrieve cached regex pattern
    kw_key = "|".join(sorted(keywords))
    if kw_key not in _pattern_cache:
        escaped = [re.escape(kw) for kw in keywords]
        _pattern_cache[kw_key] = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

    pattern = _pattern_cache[kw_key]
    query_lower = query.lower()

    matches = pattern.findall(query_lower)
    unique_matches = list(set(m.lower() for m in matches))

    # Confidence: scale by how many distinct keywords matched
    # 1 keyword → 0.6, 2 → 0.8, 3+ → 1.0
    if len(unique_matches) == 0:
        confidence = 0.0
    elif len(unique_matches) == 1:
        confidence = 0.6
    elif len(unique_matches) == 2:
        confidence = 0.8
    else:
        confidence = 1.0

    has_intent = len(unique_matches) > 0

    if has_intent:
        logger.info(
            "Visual intent detected: %s (confidence=%.1f, keywords=%s)",
            query[:60], confidence, unique_matches,
        )

    return {
        "has_visual_intent": has_intent,
        "confidence": confidence,
        "matched_keywords": unique_matches,
    }
