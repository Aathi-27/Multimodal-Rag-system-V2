"""
Query Cost Tracker — Per-query resource accounting.

Tracks for every query:
  - Prompt tokens consumed
  - Completion tokens generated
  - GPU time (retrieval + generation)
  - Estimated cost (based on local compute amortization)

Provides aggregate cost-per-1K-queries for business reporting.

Business metric: Speaks the language of cost efficiency,
demonstrates responsible resource usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Amortized local cost estimates (based on GTX 1650 power draw)
# ~50W GPU * electricity ~$0.12/kWh ≈ $0.006/hour GPU time
GPU_COST_PER_SECOND = 0.006 / 3600  # ≈ $0.0000017/sec
CPU_COST_PER_SECOND = 0.003 / 3600  # Rough estimate for CPU


@dataclass
class QueryCost:
    """Cost breakdown for a single query."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    total_time_s: float = 0.0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retrieval_time_ms": round(self.retrieval_time_s * 1000, 1),
            "generation_time_ms": round(self.generation_time_s * 1000, 1),
            "total_time_ms": round(self.total_time_s * 1000, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class CostTracker:
    """Aggregates query costs for business reporting."""

    def __init__(self) -> None:
        self._total_queries: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_gpu_time: float = 0.0

    def compute_query_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        retrieval_time_s: float,
        generation_time_s: float,
        total_time_s: float,
    ) -> QueryCost:
        """Compute cost for a single query and update aggregates."""
        total_tokens = prompt_tokens + completion_tokens

        # GPU cost = generation time (GPU-bound) + embedding time (GPU-bound)
        gpu_cost = generation_time_s * GPU_COST_PER_SECOND
        # CPU cost = retrieval + reranking
        cpu_cost = retrieval_time_s * CPU_COST_PER_SECOND
        estimated_cost = gpu_cost + cpu_cost

        cost = QueryCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            retrieval_time_s=retrieval_time_s,
            generation_time_s=generation_time_s,
            total_time_s=total_time_s,
            estimated_cost_usd=estimated_cost,
        )

        # Update aggregates
        self._total_queries += 1
        self._total_tokens += total_tokens
        self._total_cost += estimated_cost
        self._total_gpu_time += generation_time_s

        return cost

    def summary(self) -> dict:
        """Aggregate cost report for business stakeholders."""
        cost_per_1k = (
            (self._total_cost / self._total_queries) * 1000
            if self._total_queries > 0
            else 0.0
        )
        return {
            "total_queries": self._total_queries,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "cost_per_1k_queries_usd": round(cost_per_1k, 4),
            "avg_tokens_per_query": (
                round(self._total_tokens / self._total_queries)
                if self._total_queries > 0
                else 0
            ),
            "total_gpu_time_s": round(self._total_gpu_time, 2),
        }
