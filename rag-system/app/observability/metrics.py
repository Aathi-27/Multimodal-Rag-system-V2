"""
Metrics - Prometheus-compatible metrics for system observability.

Tracks:
- Upload counts and latency by modality
- Query latency (end-to-end and per-stage)
- Retrieval quality (hit counts, RRF scores)
- Model inference latency
- Corpus size
- Active worker counts
"""

from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    description: str
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, label: str = "") -> None:
        self._value += amount
        if label:
            self._labels[label] = self._labels.get(label, 0.0) + amount

    @property
    def value(self) -> float:
        return self._value

    def get(self, label: str = "") -> float:
        if label:
            return self._labels.get(label, 0.0)
        return self._value


@dataclass
class Histogram:
    """Simple histogram metric for latency tracking."""
    name: str
    description: str
    _observations: list[float] = field(default_factory=list)
    _label_observations: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, value: float, label: str = "") -> None:
        self._observations.append(value)
        if label:
            self._label_observations.setdefault(label, []).append(value)

    @property
    def count(self) -> int:
        return len(self._observations)

    @property
    def avg(self) -> float:
        if not self._observations:
            return 0.0
        return sum(self._observations) / len(self._observations)

    def p50(self) -> float:
        if not self._observations:
            return 0.0
        sorted_obs = sorted(self._observations)
        idx = int(len(sorted_obs) * 0.50)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def p95(self) -> float:
        if not self._observations:
            return 0.0
        sorted_obs = sorted(self._observations)
        idx = int(len(sorted_obs) * 0.95)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]


@dataclass
class Gauge:
    """Simple gauge metric."""
    name: str
    description: str
    _value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class MetricsCollector:
    """Central metrics collector for the RAG system."""

    def __init__(self) -> None:
        # Upload metrics
        self.uploads_total = Counter("uploads_total", "Total file uploads")
        self.upload_errors = Counter("upload_errors", "Upload errors by type")
        self.upload_latency = Histogram("upload_latency_seconds", "Upload processing latency")

        # Query metrics
        self.queries_total = Counter("queries_total", "Total queries processed")
        self.query_latency = Histogram("query_latency_seconds", "End-to-end query latency")
        self.retrieval_latency = Histogram("retrieval_latency_seconds", "Retrieval stage latency")
        self.rerank_latency = Histogram("rerank_latency_seconds", "Reranking latency")
        self.generation_latency = Histogram("generation_latency_seconds", "LLM generation latency")

        # Retrieval quality
        self.retrieval_hits = Histogram("retrieval_hits", "Number of hits per query")
        self.rrf_top_score = Histogram("rrf_top_score", "Top RRF score per query")

        # System gauges
        self.corpus_size = Gauge("corpus_size", "Total chunks in corpus")
        self.active_workers = Gauge("active_workers", "Active ingestion workers")
        self.model_loaded = Gauge("models_loaded", "Number of loaded models")

    @contextmanager
    def track_latency(self, histogram: Histogram, label: str = "") -> Generator[None, None, None]:
        """Context manager to automatically track latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            histogram.observe(elapsed, label)

    def snapshot(self) -> dict:
        """Return a snapshot of all metrics for the /health endpoint."""
        return {
            "uploads": {
                "total": self.uploads_total.value,
                "errors": self.upload_errors.value,
                "avg_latency_s": round(self.upload_latency.avg, 3),
            },
            "queries": {
                "total": self.queries_total.value,
                "avg_latency_s": round(self.query_latency.avg, 3),
                "p95_latency_s": round(self.query_latency.p95(), 3),
            },
            "corpus_size": int(self.corpus_size.value),
            "active_workers": int(self.active_workers.value),
        }


# Singleton instance
metrics = MetricsCollector()
