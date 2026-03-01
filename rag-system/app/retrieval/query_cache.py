"""
Query Result Cache — LRU cache for frequently repeated queries.

Reduces latency for repeated or near-identical queries by caching
retrieval + generation results. Configurable TTL and max size.

Business metric: Reduces P95 latency, lowers compute cost per query.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached query result."""
    query_hash: str
    query_text: str
    results: list[dict]
    answer: str
    citations: list[dict]
    confidence: dict
    cost: dict
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


class QueryCache:
    """
    Thread-safe LRU query cache with TTL expiration.

    Cache key = normalized query text hash.
    On cache hit, skips retrieval + LLM generation entirely.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_query(query: str) -> str:
        """Deterministic hash of normalized query text."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[CacheEntry]:
        """Look up a cached result. Returns None on miss or expired."""
        key = self._hash_query(query)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(self._ttl):
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache expired for query: %s", query[:60])
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            logger.info("Cache HIT for query: %s (hits=%d)", query[:60], entry.hit_count)
            return entry

    def put(
        self,
        query: str,
        results: list[dict],
        answer: str,
        citations: list[dict],
        confidence: dict | None = None,
        cost: dict | None = None,
    ) -> None:
        """Store a query result in the cache."""
        key = self._hash_query(query)
        entry = CacheEntry(
            query_hash=key,
            query_text=query,
            results=results,
            answer=answer,
            citations=citations,
            confidence=confidence or {},
            cost=cost or {},
        )
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = entry
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Cache evicted: %s", evicted_key)

    def invalidate(self, query: str | None = None) -> int:
        """Invalidate a specific query or the entire cache."""
        with self._lock:
            if query:
                key = self._hash_query(query)
                if key in self._cache:
                    del self._cache[key]
                    return 1
                return 0
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> dict:
        """Cache performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "total_lookups": total,
            }
