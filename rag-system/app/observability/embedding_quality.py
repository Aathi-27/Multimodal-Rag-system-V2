"""
Embedding Quality Checker — Monitor embedding norms and similarity distributions.

Provides diagnostics on the quality of stored embeddings:
  - Average L2 norm (should be ~1.0 for normalized embeddings)
  - Norm variance (high variance → inconsistent normalization)
  - Intra-corpus cosine similarity distribution
  - Outlier detection (embeddings with abnormal norms)
  - Per-source embedding statistics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingQualityChecker:
    """Monitor and analyze the quality of stored embeddings."""

    def __init__(self, vector_store, embedding_model) -> None:
        self._vector_store = vector_store
        self._embedding_model = embedding_model

    def check(self, sample_size: int = 200) -> dict:
        """
        Run embedding quality diagnostics on a sample of stored vectors.

        Args:
            sample_size: Number of vectors to sample for analysis.

        Returns:
            avg_norm            — Mean L2 norm across sampled vectors
            norm_std            — Standard deviation of norms
            norm_min / norm_max — Range of norms
            outlier_count       — Vectors with norm >2σ from mean
            cosine_stats        — { mean, std, min, max, p25, p75 }
            embedding_dim       — Dimensionality
            total_vectors       — Total vectors in collection
            sample_analyzed     — Actual number analyzed
            health_status       — "good" / "warning" / "critical"
            issues              — List of detected issues
        """
        client = self._vector_store.client
        collection = self._vector_store.collection_name

        # Get total count
        total_vectors = self._vector_store.count()
        if total_vectors == 0:
            return {
                "health_status": "critical",
                "issues": ["No vectors in collection"],
                "total_vectors": 0,
                "sample_analyzed": 0,
            }

        # Sample vectors
        actual_sample = min(sample_size, total_vectors)
        points = []
        offset = None
        remaining = actual_sample

        while remaining > 0:
            batch_size = min(remaining, 100)
            result = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            batch_points, next_offset = result
            points.extend(batch_points)
            remaining -= len(batch_points)
            if next_offset is None:
                break
            offset = next_offset

        if not points:
            return {
                "health_status": "critical",
                "issues": ["Failed to retrieve any vectors"],
                "total_vectors": total_vectors,
                "sample_analyzed": 0,
            }

        # Extract vectors as numpy array
        vectors = np.array([p.vector for p in points], dtype=np.float32)
        sample_count = len(vectors)
        dim = vectors.shape[1]

        # ── Norm Analysis ─────────────────────────────────────────────
        norms = np.linalg.norm(vectors, axis=1)
        avg_norm = float(np.mean(norms))
        norm_std = float(np.std(norms))
        norm_min = float(np.min(norms))
        norm_max = float(np.max(norms))

        # Outliers: >2σ from mean
        outlier_mask = np.abs(norms - avg_norm) > 2 * norm_std
        outlier_count = int(np.sum(outlier_mask))

        # ── Cosine Similarity Analysis ────────────────────────────────
        # Compute pairwise cosine similarities on a subset
        cosine_sample = min(100, sample_count)
        subset = vectors[:cosine_sample]

        # Normalize for cosine similarity
        norms_sub = np.linalg.norm(subset, axis=1, keepdims=True)
        norms_sub = np.where(norms_sub == 0, 1, norms_sub)
        normalized = subset / norms_sub

        # Pairwise cosine similarity matrix
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (excluding diagonal)
        upper_idx = np.triu_indices(cosine_sample, k=1)
        pairwise_sims = sim_matrix[upper_idx]

        cosine_stats = {}
        if len(pairwise_sims) > 0:
            cosine_stats = {
                "mean": round(float(np.mean(pairwise_sims)), 4),
                "std": round(float(np.std(pairwise_sims)), 4),
                "min": round(float(np.min(pairwise_sims)), 4),
                "max": round(float(np.max(pairwise_sims)), 4),
                "p25": round(float(np.percentile(pairwise_sims, 25)), 4),
                "p75": round(float(np.percentile(pairwise_sims, 75)), 4),
                "pairs_analyzed": len(pairwise_sims),
            }

        # ── Per-source analysis ───────────────────────────────────────
        source_stats: dict[str, list[float]] = {}
        for p, norm_val in zip(points, norms):
            source = (p.payload or {}).get("source", "unknown")
            if source not in source_stats:
                source_stats[source] = []
            source_stats[source].append(float(norm_val))

        per_source = [
            {
                "source": src,
                "count": len(norms_list),
                "avg_norm": round(float(np.mean(norms_list)), 4),
                "norm_std": round(float(np.std(norms_list)), 4) if len(norms_list) > 1 else 0.0,
            }
            for src, norms_list in source_stats.items()
        ]
        per_source.sort(key=lambda x: x["count"], reverse=True)

        # ── Health Assessment ─────────────────────────────────────────
        issues: list[str] = []

        # Check norm normalization (BGE-small should produce ~1.0 norms)
        if abs(avg_norm - 1.0) > 0.1:
            issues.append(f"Average norm ({avg_norm:.4f}) deviates from expected 1.0")
        if norm_std > 0.15:
            issues.append(f"High norm variance (std={norm_std:.4f})")
        if outlier_count > sample_count * 0.05:
            issues.append(f"{outlier_count} outlier vectors detected ({outlier_count/sample_count*100:.1f}%)")

        # Check cosine similarity distribution
        if cosine_stats:
            if cosine_stats["mean"] > 0.8:
                issues.append(f"Very high avg cosine similarity ({cosine_stats['mean']:.4f}) — possible redundancy")
            elif cosine_stats["mean"] < 0.1:
                issues.append(f"Very low avg cosine similarity ({cosine_stats['mean']:.4f}) — embeddings may be poor")

        if not issues:
            health = "good"
        elif len(issues) <= 2:
            health = "warning"
        else:
            health = "critical"

        return {
            "health_status": health,
            "issues": issues,
            "total_vectors": total_vectors,
            "sample_analyzed": sample_count,
            "embedding_dim": dim,
            "norm_stats": {
                "avg": round(avg_norm, 4),
                "std": round(norm_std, 4),
                "min": round(norm_min, 4),
                "max": round(norm_max, 4),
                "outlier_count": outlier_count,
            },
            "cosine_stats": cosine_stats,
            "per_source": per_source,
        }
