"""
Tracing - OpenTelemetry-compatible distributed tracing setup.

Provides request-level tracing across all pipeline stages:
  Ingestion → Processing → Indexing → Retrieval → Reranking → Generation

Each trace includes:
- Upload ID / Query ID as trace context
- Per-stage spans with timing
- Error recording
"""

from __future__ import annotations

import uuid
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single span within a trace."""
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict = field(default_factory=dict)
    status: str = "OK"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """A complete trace representing a request lifecycle."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return (end - start) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "spans": [
                {
                    "name": s.name,
                    "span_id": s.span_id,
                    "duration_ms": round(s.duration_ms, 2),
                    "status": s.status,
                    "error": s.error,
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
            "metadata": self.metadata,
        }


class Tracer:
    """Lightweight tracer for pipeline stage tracking."""

    def __init__(self) -> None:
        self._active_traces: dict[str, Trace] = {}

    def start_trace(self, metadata: Optional[dict] = None) -> Trace:
        """Start a new trace."""
        trace = Trace(metadata=metadata or {})
        self._active_traces[trace.trace_id] = trace
        logger.debug("Started trace: %s", trace.trace_id)
        return trace

    @contextmanager
    def span(
        self,
        trace: Trace,
        name: str,
        attributes: Optional[dict] = None,
    ) -> Generator[Span, None, None]:
        """Context manager that creates and records a span."""
        span = Span(
            name=name,
            trace_id=trace.trace_id,
            start_time=time.perf_counter(),
            attributes=attributes or {},
        )
        try:
            yield span
        except Exception as e:
            span.status = "ERROR"
            span.error = str(e)
            raise
        finally:
            span.end_time = time.perf_counter()
            trace.add_span(span)
            logger.debug(
                "Span [%s] %s: %.2fms",
                trace.trace_id[:8],
                name,
                span.duration_ms,
            )

    def finish_trace(self, trace_id: str) -> Optional[Trace]:
        """Finish and remove a trace from active tracking."""
        return self._active_traces.pop(trace_id, None)


# Singleton instance
tracer = Tracer()
