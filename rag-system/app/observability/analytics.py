"""
Retrieval Analytics Tracker - Lightweight per-document query statistics.

Tracks:
- Retrieval count per source document
- Last queried timestamp
- Average reranker score per source
- Average rank position per source

Persists to a JSON file so stats survive restarts.
Thread-safe via a lock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AnalyticsTracker:
    """Accumulates per-document retrieval statistics."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "retrieval_analytics.json"
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load analytics from disk."""
        if self._file.exists():
            try:
                with open(self._file, "r") as f:
                    self._stats = json.load(f)
                logger.info("Analytics loaded: %d documents tracked.", len(self._stats))
            except Exception as e:
                logger.warning("Failed to load analytics: %s", e)
                self._stats = {}
        else:
            self._stats = {}

    def _save(self) -> None:
        """Persist analytics to disk (call inside lock)."""
        try:
            with open(self._file, "w") as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save analytics: %s", e)

    def record_retrieval(self, results: list[dict]) -> None:
        """
        Record a retrieval event for all source documents that appeared in results.

        Args:
            results: Reranked results list, each with metadata.source, reranker_score, rrf_rank.
        """
        with self._lock:
            now = time.time()
            for result in results:
                meta = result.get("metadata", {})
                source = meta.get("source", "unknown")
                if source == "unknown":
                    continue

                if source not in self._stats:
                    self._stats[source] = {
                        "retrieval_count": 0,
                        "last_queried": 0,
                        "total_reranker_score": 0.0,
                        "total_rrf_rank": 0,
                        "hit_count": 0,  # individual chunk hits
                    }

                entry = self._stats[source]
                entry["retrieval_count"] += 1
                entry["last_queried"] = now
                entry["hit_count"] += 1
                entry["total_reranker_score"] += result.get("reranker_score", 0.0)
                entry["total_rrf_rank"] += result.get("rrf_rank", 0)

            self._save()

    def get_stats(self, source: Optional[str] = None) -> dict:
        """
        Get computed analytics.

        If source is given, return stats for that document.
        Otherwise, return stats for all documents.
        """
        with self._lock:
            if source:
                raw = self._stats.get(source, {})
                return self._compute_stats(source, raw) if raw else {}

            return {
                src: self._compute_stats(src, raw)
                for src, raw in self._stats.items()
            }

    @staticmethod
    def _compute_stats(source: str, raw: dict) -> dict:
        """Compute human-readable stats from raw counters."""
        hits = raw.get("hit_count", 0)
        return {
            "source": source,
            "retrieval_count": raw.get("retrieval_count", 0),
            "last_queried": raw.get("last_queried", 0),
            "avg_reranker_score": (
                raw.get("total_reranker_score", 0) / hits if hits > 0 else 0.0
            ),
            "avg_rank_position": (
                raw.get("total_rrf_rank", 0) / hits if hits > 0 else 0
            ),
        }

    def delete_source(self, source: str) -> None:
        """Remove analytics for a deleted document."""
        with self._lock:
            self._stats.pop(source, None)
            self._save()
