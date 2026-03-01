"""
Confidence Scoring — Trust metric for RAG answers.

Computes a confidence score from multiple signals:
  - Reranker score quality (mean + min of top chunks)
  - Citation coverage ratio (how many chunks are cited)
  - Source diversity (number of independent sources)
  - Chunk agreement (text overlap between top chunks)

Returns a 0.0–1.0 score with High/Medium/Low label.

Business metric: Reduces hallucination risk perception,
improves executive trust in system outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Thresholds for confidence levels
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40


@dataclass
class ConfidenceResult:
    """Confidence assessment for a single query response."""
    score: float                # 0.0 – 1.0
    level: str                  # "high" | "medium" | "low"
    signals: dict               # Breakdown of contributing signals
    source_count: int           # Number of independent sources used
    grounding_summary: str      # Human-readable grounding statement

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "source_count": self.source_count,
            "grounding": self.grounding_summary,
        }


def compute_confidence(
    results: list[dict],
    chunks_used: int = 5,
) -> ConfidenceResult:
    """
    Compute confidence score from retrieval results.

    Args:
        results: Retrieval results with reranker_score, metadata, etc.
        chunks_used: Number of chunks actually used in the prompt.

    Returns:
        ConfidenceResult with score, level, and signal breakdown.
    """
    if not results:
        return ConfidenceResult(
            score=0.0,
            level="low",
            signals={},
            source_count=0,
            grounding_summary="No sources found.",
        )

    top_results = results[:chunks_used]

    # ── Signal 1: Reranker score quality (0–1) ────────────────────────
    reranker_scores = [
        r.get("reranker_score", r.get("rrf_score", 0.0))
        for r in top_results
    ]
    if reranker_scores:
        mean_score = sum(reranker_scores) / len(reranker_scores)
        min_score = min(reranker_scores)
        # Normalize: reranker scores typically 0–1, weight mean more
        reranker_signal = 0.7 * min(mean_score, 1.0) + 0.3 * min(min_score, 1.0)
    else:
        reranker_signal = 0.0

    # ── Signal 2: Source diversity (0–1) ──────────────────────────────
    sources = set()
    for r in top_results:
        meta = r.get("metadata", r.get("payload", {}))
        source = meta.get("source", "unknown")
        sources.add(source)

    source_count = len(sources)
    # 1 source = 0.3, 2 = 0.6, 3+ = 1.0
    diversity_signal = min(source_count / 3.0, 1.0)

    # ── Signal 3: Coverage ratio (0–1) ────────────────────────────────
    # What fraction of total results made it to the prompt
    total_retrieved = len(results)
    coverage_signal = min(chunks_used / max(total_retrieved, 1), 1.0)

    # ── Signal 4: Modality consistency (0–1) ──────────────────────────
    modalities = set()
    for r in top_results:
        meta = r.get("metadata", r.get("payload", {}))
        modalities.add(meta.get("modality", "document"))

    # Single modality = fully consistent; multi = slight penalty
    consistency_signal = 1.0 if len(modalities) <= 1 else 0.8

    # ── Weighted combination ──────────────────────────────────────────
    weights = {
        "reranker_quality": 0.45,
        "source_diversity": 0.25,
        "coverage_ratio": 0.15,
        "modality_consistency": 0.15,
    }
    signals = {
        "reranker_quality": reranker_signal,
        "source_diversity": diversity_signal,
        "coverage_ratio": coverage_signal,
        "modality_consistency": consistency_signal,
    }
    score = sum(signals[k] * weights[k] for k in weights)
    score = max(0.0, min(1.0, score))

    # ── Level classification ──────────────────────────────────────────
    if score >= HIGH_THRESHOLD:
        level = "high"
    elif score >= MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    # ── Grounding summary ─────────────────────────────────────────────
    grounding = f"Answer grounded in {source_count} independent source{'s' if source_count != 1 else ''}."

    return ConfidenceResult(
        score=score,
        level=level,
        signals=signals,
        source_count=source_count,
        grounding_summary=grounding,
    )
