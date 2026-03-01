"""
Reranker - Cross-encoder reranking via BGE-reranker-base.

Rules (from PRD):
- Use BAAI/bge-reranker-base loaded via sentence-transformers
- Apply to top 15 RRF candidates only
- Drop results with score < 0.15 (normalized)
- Retain minimum 5 results as safeguard
- Target latency: < 50ms for 15 candidates
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using BGE-reranker-base."""

    def __init__(self, model_manager) -> None:
        self._manager = model_manager

    @property
    def _model(self):
        return self._manager.get_reranker_model()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        threshold: float = 0.15,
        min_results: int = 5,
    ) -> list[dict]:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            query: The user's query.
            candidates: RRF-fused candidates (max 15).
            threshold: Minimum score to keep (normalized).
            min_results: Minimum results to retain (safeguard).

        Returns:
            Reranked list sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = []
        for candidate in candidates:
            chunk_text = candidate.get("metadata", {}).get("text", "")
            if not chunk_text:
                chunk_text = str(candidate.get("metadata", {}))
            pairs.append((query, chunk_text))

        # Score with cross-encoder
        scores = self._model.predict(pairs)

        # Normalize scores to [0, 1] range using sigmoid
        import numpy as np
        normalized_scores = 1.0 / (1.0 + np.exp(-np.array(scores)))

        # Attach scores to candidates
        scored = []
        for candidate, score in zip(candidates, normalized_scores):
            candidate_copy = candidate.copy()
            candidate_copy["reranker_score"] = float(score)
            scored.append(candidate_copy)

        # Sort by reranker score descending
        scored.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Apply threshold with min_results safeguard
        above_threshold = [c for c in scored if c["reranker_score"] >= threshold]

        if len(above_threshold) >= min_results:
            result = above_threshold
        else:
            # Keep at least min_results even if below threshold
            result = scored[:max(min_results, len(above_threshold))]

        logger.info(
            "Reranked %d → %d candidates (threshold=%.2f, min=%d)",
            len(candidates),
            len(result),
            threshold,
            min_results,
        )

        return result
