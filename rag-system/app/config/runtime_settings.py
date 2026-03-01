"""
Runtime Settings - Mutable in-memory overrides for retrieval parameters.

The base settings are loaded from config.yaml at startup via @lru_cache.
This module provides a thin mutable layer that can be patched at runtime
through the /settings/retrieval API without restarting the server.

Overrides are stored in a plain dict.  Any key present in the overrides
dict takes precedence over the frozen Settings dataclass.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Only these keys are allowed as runtime overrides
ALLOWED_KEYS = frozenset({
    "vector_top_k",
    "bm25_top_k",
    "rrf_k",
    "rerank_count",
    "rerank_threshold",
    "rerank_min_results",
})

# Validation: (min, max, type)
KEY_CONSTRAINTS: dict[str, tuple[type, float, float]] = {
    "vector_top_k":       (int,   1,   200),
    "bm25_top_k":         (int,   1,   200),
    "rrf_k":              (int,   1,   200),
    "rerank_count":       (int,   1,   100),
    "rerank_threshold":   (float, 0.0, 1.0),
    "rerank_min_results": (int,   1,   50),
}

_overrides: dict[str, Any] = {}


def get_overrides() -> dict[str, Any]:
    """Return a snapshot of current runtime overrides."""
    with _lock:
        return dict(_overrides)


def set_overrides(patch: dict[str, Any]) -> dict[str, str]:
    """
    Merge *patch* into runtime overrides.

    Returns a dict of validation errors (empty = success).
    """
    errors: dict[str, str] = {}
    with _lock:
        for key, value in patch.items():
            if key not in ALLOWED_KEYS:
                errors[key] = f"Unknown key. Allowed: {sorted(ALLOWED_KEYS)}"
                continue
            expected_type, lo, hi = KEY_CONSTRAINTS[key]
            # Coerce
            try:
                value = expected_type(value)
            except (ValueError, TypeError):
                errors[key] = f"Expected {expected_type.__name__}, got {type(value).__name__}"
                continue
            if not (lo <= value <= hi):
                errors[key] = f"Value {value} out of range [{lo}, {hi}]"
                continue
            _overrides[key] = value
            logger.info("Runtime override: %s = %s", key, value)
    return errors


def reset_overrides() -> None:
    """Clear all runtime overrides (revert to config.yaml defaults)."""
    with _lock:
        _overrides.clear()
    logger.info("All runtime overrides cleared.")


def get_effective_retrieval_params() -> dict[str, Any]:
    """
    Return the effective retrieval parameters: base config + runtime overrides.

    Useful for display in the UI / debug panel.
    """
    from app.config.settings import get_settings

    base = get_settings().retrieval
    ovr = get_overrides()
    return {
        "vector_top_k":       ovr.get("vector_top_k", base.vector_top_k),
        "bm25_top_k":         ovr.get("bm25_top_k", base.bm25_top_k),
        "rrf_k":              ovr.get("rrf_k", base.rrf_k),
        "rerank_count":       ovr.get("rerank_count", base.rerank_count),
        "rerank_threshold":   ovr.get("rerank_threshold", base.rerank_threshold),
        "rerank_min_results": ovr.get("rerank_min_results", base.rerank_min_results),
    }
