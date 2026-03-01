"""
Business Impact Estimator — ROI & Time-Saved Metrics.

Converts raw system metrics into business-ready impact numbers:
  - Time saved per query  (vs manual document search)
  - ROI per 1K queries    (time value minus compute cost)
  - Productivity multiplier

Assumptions (configurable):
  - Average manual search time: 4 minutes per question
  - Average employee hourly cost: $35/hr
  - These are conservative estimates for knowledge worker tasks

Business metric:  "Each query saves 4 minutes of manual search"
                  "ROI: $2.31 per 1K queries"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Configurable assumptions ──────────────────────────────────────────────────
MANUAL_SEARCH_SECONDS = 240       # 4 minutes average for manual doc search
EMPLOYEE_HOURLY_COST_USD = 35.0   # Conservative knowledge worker rate
EMPLOYEE_COST_PER_SECOND = EMPLOYEE_HOURLY_COST_USD / 3600  # ~$0.0097/sec


@dataclass
class BusinessImpact:
    """Business impact assessment for the system."""
    # Per-query
    manual_search_time_s: float       # How long manual search would take
    system_response_time_s: float     # How long RAG took
    time_saved_s: float               # Delta
    time_saved_pct: float             # Percentage improvement

    # Per 1K queries
    total_time_saved_per_1k_s: float
    labor_saved_per_1k_usd: float     # Value of time saved
    compute_cost_per_1k_usd: float    # Actual compute cost
    net_roi_per_1k_usd: float         # Labor saved minus compute cost
    roi_multiplier: float             # Return / Cost ratio

    # Productivity
    speedup_factor: float             # manual_time / system_time

    def to_dict(self) -> dict:
        return {
            "per_query": {
                "manual_search_time_s": round(self.manual_search_time_s, 1),
                "system_response_time_s": round(self.system_response_time_s, 1),
                "time_saved_s": round(self.time_saved_s, 1),
                "time_saved_pct": round(self.time_saved_pct, 1),
            },
            "per_1k_queries": {
                "total_time_saved_hours": round(self.total_time_saved_per_1k_s / 3600, 1),
                "labor_saved_usd": round(self.labor_saved_per_1k_usd, 2),
                "compute_cost_usd": round(self.compute_cost_per_1k_usd, 4),
                "net_roi_usd": round(self.net_roi_per_1k_usd, 2),
                "roi_multiplier": f"{self.roi_multiplier:.0f}x",
            },
            "speedup_factor": f"{self.speedup_factor:.1f}x",
        }


def estimate_business_impact(
    avg_response_time_s: float,
    compute_cost_per_1k_usd: float,
    manual_search_time_s: float = MANUAL_SEARCH_SECONDS,
) -> BusinessImpact:
    """
    Estimate business impact of RAG system vs manual document search.

    Args:
        avg_response_time_s:    Average system response time (seconds)
        compute_cost_per_1k_usd: Compute cost per 1K queries
        manual_search_time_s:   Assumed manual search time per question

    Returns:
        BusinessImpact with ROI, time savings, and productivity metrics
    """
    time_saved_s = max(0, manual_search_time_s - avg_response_time_s)
    time_saved_pct = (time_saved_s / manual_search_time_s * 100) if manual_search_time_s > 0 else 0

    total_time_saved_per_1k = time_saved_s * 1000
    labor_saved_per_1k = total_time_saved_per_1k * EMPLOYEE_COST_PER_SECOND
    net_roi = labor_saved_per_1k - compute_cost_per_1k_usd
    roi_multiplier = labor_saved_per_1k / compute_cost_per_1k_usd if compute_cost_per_1k_usd > 0 else float("inf")

    speedup = manual_search_time_s / avg_response_time_s if avg_response_time_s > 0 else float("inf")

    return BusinessImpact(
        manual_search_time_s=manual_search_time_s,
        system_response_time_s=avg_response_time_s,
        time_saved_s=time_saved_s,
        time_saved_pct=time_saved_pct,
        total_time_saved_per_1k_s=total_time_saved_per_1k,
        labor_saved_per_1k_usd=labor_saved_per_1k,
        compute_cost_per_1k_usd=compute_cost_per_1k_usd,
        net_roi_per_1k_usd=net_roi,
        roi_multiplier=roi_multiplier,
        speedup_factor=speedup,
    )
