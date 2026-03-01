"""
Corpus Coverage Analyzer — Per-chunk retrieval statistics and redundancy detection.

Analyzes query history to compute:
  - Per-chunk retrieval frequency (how often each chunk appears in results)
  - Never-retrieved chunks (corpus dead zones)
  - Frequently-retrieved chunks (hotspots)
  - Avg rank and reranker score per chunk
  - Redundancy clusters (chunks with very similar retrieval patterns)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class CorpusCoverageAnalyzer:
    """Analyzes how thoroughly the corpus is being utilized by queries."""

    def __init__(self, query_store, vector_store) -> None:
        self._query_store = query_store
        self._vector_store = vector_store

    def analyze(self, max_queries: int = 500) -> dict:
        """
        Run full corpus coverage analysis.

        Returns:
            total_corpus_chunks     — Total chunks in the index
            total_queries_analyzed  — Number of queries scanned
            coverage_rate           — Fraction of corpus ever retrieved
            never_retrieved         — List of chunk IDs never seen in results
            never_retrieved_count   — Count of never-retrieved chunks
            hotspot_chunks          — Most frequently retrieved chunks
            coldspot_chunks         — Least frequently retrieved chunks (but >0)
            per_source_coverage     — Coverage breakdown by document source
            chunk_stats             — Per-chunk retrieval statistics
        """
        # Get all chunks from the index
        all_docs = self._vector_store.list_documents()
        all_sources: dict[str, int] = {d["source"]: d["chunk_count"] for d in all_docs}
        total_corpus_chunks = sum(all_sources.values())

        # Get all chunk IDs
        all_chunk_ids = set()
        for source in all_sources:
            chunks = self._vector_store.get_chunks_by_source(source)
            for c in chunks:
                all_chunk_ids.add(c["chunk_id"])

        # Scan query history for retrieval patterns
        records, total = self._query_store.get_all(limit=max_queries, offset=0)

        chunk_stats: dict[str, dict] = defaultdict(lambda: {
            "retrieval_count": 0,
            "total_rank": 0,
            "total_score": 0.0,
            "sources_set": set(),
        })

        for record in records:
            retrieved = record.get("retrieved_chunks", [])
            for i, chunk in enumerate(retrieved):
                cid = chunk.get("chunk_id", "")
                if not cid:
                    continue
                stats = chunk_stats[cid]
                stats["retrieval_count"] += 1
                stats["total_rank"] += (i + 1)
                stats["total_score"] += chunk.get("score", 0)
                stats["sources_set"].add(chunk.get("source", "unknown"))

            # Also check survival logs
            survival_log = record.get("survival_log") or []
            for entry in survival_log:
                cid = entry.get("chunk_id", "")
                if cid and entry.get("survived"):
                    # Already counted above from retrieved_chunks
                    pass

        # Compute per-chunk metrics
        retrieved_ids = set(chunk_stats.keys())
        never_retrieved = all_chunk_ids - retrieved_ids

        chunk_metrics: list[dict] = []
        for cid, stats in chunk_stats.items():
            count = stats["retrieval_count"]
            chunk_metrics.append({
                "chunk_id": cid,
                "retrieval_count": count,
                "avg_rank": round(stats["total_rank"] / count, 2) if count > 0 else 0,
                "avg_score": round(stats["total_score"] / count, 4) if count > 0 else 0,
            })

        chunk_metrics.sort(key=lambda x: x["retrieval_count"], reverse=True)

        # Hotspots (top 10 most retrieved)
        hotspots = chunk_metrics[:10]

        # Coldspots (bottom 10 with at least 1 retrieval)
        coldspots = sorted(
            [c for c in chunk_metrics if c["retrieval_count"] > 0],
            key=lambda x: x["retrieval_count"],
        )[:10]

        # Per-source coverage
        source_coverage: list[dict] = []
        for source, total_chunks in all_sources.items():
            source_chunks = self._vector_store.get_chunks_by_source(source)
            source_chunk_ids = {c["chunk_id"] for c in source_chunks}
            retrieved_from_source = source_chunk_ids & retrieved_ids
            coverage = len(retrieved_from_source) / total_chunks if total_chunks > 0 else 0

            source_coverage.append({
                "source": source,
                "total_chunks": total_chunks,
                "retrieved_chunks": len(retrieved_from_source),
                "coverage_rate": round(coverage, 4),
                "never_retrieved": total_chunks - len(retrieved_from_source),
            })

        source_coverage.sort(key=lambda x: x["coverage_rate"])

        coverage_rate = len(retrieved_ids) / total_corpus_chunks if total_corpus_chunks > 0 else 0

        return {
            "total_corpus_chunks": total_corpus_chunks,
            "total_queries_analyzed": len(records),
            "coverage_rate": round(coverage_rate, 4),
            "retrieved_chunk_count": len(retrieved_ids),
            "never_retrieved_count": len(never_retrieved),
            "never_retrieved_sample": list(never_retrieved)[:20],
            "hotspot_chunks": hotspots,
            "coldspot_chunks": coldspots,
            "per_source_coverage": source_coverage,
        }
