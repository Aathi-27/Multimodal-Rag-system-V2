"""
Experiment Engine — Side-by-side comparison and batch evaluation.

Enables controlled experimentation:
  1. Side-by-Side Comparison
     Run the same query through the pipeline with two different parameter sets
     and compare retrieval quality (overlap, rank differences, unique finds).

  2. Batch Evaluation
     Run a test dataset of queries (with optional ground truth) and compute
     aggregate metrics: recall@k, MRR, avg latency, survival rates.

  3. Parameter Sweep (future)
     Automatically test parameter ranges and find optimal settings.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    query: str
    params: dict
    chunks: list[dict]
    chunk_count: int
    retrieval_latency: float
    survival_log: list[dict]

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "query": self.query,
            "params": self.params,
            "chunks": self.chunks,
            "chunk_count": self.chunk_count,
            "retrieval_latency": self.retrieval_latency,
            "survival_log": self.survival_log,
        }


class ExperimentEngine:
    """Run controlled retrieval experiments."""

    def __init__(self, hybrid_retriever, embedding_model) -> None:
        self._retriever = hybrid_retriever
        self._embedding_model = embedding_model

    def run_single(
        self,
        query: str,
        overrides: dict | None = None,
    ) -> ExperimentResult:
        """
        Run a single query through the pipeline with optional parameter overrides.

        Args:
            query: Query text.
            overrides: Parameter overrides (vector_top_k, bm25_top_k, etc.)

        Returns:
            ExperimentResult with chunks, latency, and survival log.
        """
        experiment_id = uuid.uuid4().hex[:12]

        # Generate embedding
        query_embedding = self._embedding_model.embed_query(query)

        # Run retrieval with debug=True to get survival data
        t_start = time.perf_counter()
        output = self._retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            debug=True,
            overrides=overrides,
        )
        t_end = time.perf_counter()

        if isinstance(output, tuple):
            results, debug_info = output
        else:
            results = output
            debug_info = {}

        # Extract survival log from debug info
        survival_log = debug_info.get("survival_log", [])

        # Build compact chunk summaries
        chunks = []
        for i, r in enumerate(results):
            meta = r.get("metadata", r.get("payload", {}))
            chunks.append({
                "chunk_id": str(r.get("chunk_id", ""))[:12],
                "source": meta.get("source", "unknown"),
                "page_start": meta.get("page_start"),
                "reranker_score": round(r.get("reranker_score", 0), 4),
                "rank": i + 1,
                "text_preview": meta.get("text", "")[:120],
            })

        return ExperimentResult(
            experiment_id=experiment_id,
            query=query,
            params=overrides or {},
            chunks=chunks,
            chunk_count=len(results),
            retrieval_latency=round(t_end - t_start, 4),
            survival_log=survival_log,
        )

    def compare(
        self,
        query: str,
        params_a: dict,
        params_b: dict,
        label_a: str = "A",
        label_b: str = "B",
    ) -> dict:
        """
        Run side-by-side comparison with two parameter sets.

        Returns:
            comparison with overlap analysis, unique chunks, rank diffs, etc.
        """
        result_a = self.run_single(query, overrides=params_a)
        result_b = self.run_single(query, overrides=params_b)

        # Compute overlap
        ids_a = {c["chunk_id"] for c in result_a.chunks}
        ids_b = {c["chunk_id"] for c in result_b.chunks}
        overlap = ids_a & ids_b
        unique_a = ids_a - ids_b
        unique_b = ids_b - ids_a

        # Rank correlation for overlapping chunks
        rank_diffs = []
        for cid in overlap:
            rank_a = next((c["rank"] for c in result_a.chunks if c["chunk_id"] == cid), 0)
            rank_b = next((c["rank"] for c in result_b.chunks if c["chunk_id"] == cid), 0)
            rank_diffs.append({
                "chunk_id": cid,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_diff": rank_b - rank_a,
            })

        # Score comparison for overlapping chunks
        score_comparison = []
        for cid in overlap:
            score_a = next((c["reranker_score"] for c in result_a.chunks if c["chunk_id"] == cid), 0)
            score_b = next((c["reranker_score"] for c in result_b.chunks if c["chunk_id"] == cid), 0)
            score_comparison.append({
                "chunk_id": cid,
                "score_a": score_a,
                "score_b": score_b,
                "score_diff": round(score_b - score_a, 4),
            })

        return {
            "query": query,
            "experiment_a": {
                "label": label_a,
                "params": params_a,
                "result": result_a.to_dict(),
            },
            "experiment_b": {
                "label": label_b,
                "params": params_b,
                "result": result_b.to_dict(),
            },
            "analysis": {
                "overlap_count": len(overlap),
                "unique_to_a": len(unique_a),
                "unique_to_b": len(unique_b),
                "jaccard_similarity": round(
                    len(overlap) / len(ids_a | ids_b), 4
                ) if (ids_a | ids_b) else 0.0,
                "rank_differences": sorted(rank_diffs, key=lambda x: abs(x["rank_diff"]), reverse=True),
                "score_comparison": score_comparison,
                "latency_diff": round(result_b.retrieval_latency - result_a.retrieval_latency, 4),
            },
        }

    def batch_evaluate(
        self,
        test_queries: list[dict],
        overrides: dict | None = None,
    ) -> dict:
        """
        Run batch evaluation on a set of test queries.

        Args:
            test_queries: List of dicts with keys:
                - query (str, required)
                - ground_truth_chunk_ids (list[str], optional)
                - expected_source (str, optional)
            overrides: Parameter overrides applied to all queries.

        Returns:
            Aggregate metrics: recall@5, recall@10, MRR, latency stats,
            plus per-query breakdown.
        """
        evaluation_id = uuid.uuid4().hex[:12]
        results: list[dict] = []

        total_recall_5 = 0.0
        total_recall_10 = 0.0
        total_mrr = 0.0
        recall_5_count = 0
        recall_10_count = 0
        mrr_count = 0
        latencies: list[float] = []

        for tq in test_queries:
            query_text = tq.get("query", "")
            if not query_text:
                continue

            ground_truth = set(tq.get("ground_truth_chunk_ids", []))
            expected_source = tq.get("expected_source")

            # Run the query
            exp_result = self.run_single(query_text, overrides=overrides)
            latencies.append(exp_result.retrieval_latency)

            retrieved_ids = [c["chunk_id"] for c in exp_result.chunks]
            retrieved_sources = [c["source"] for c in exp_result.chunks]

            # Compute metrics
            query_metrics: dict = {
                "query": query_text,
                "chunk_count": exp_result.chunk_count,
                "retrieval_latency": exp_result.retrieval_latency,
            }

            if ground_truth:
                # Recall@5
                r5 = len(ground_truth & set(retrieved_ids[:5])) / len(ground_truth)
                total_recall_5 += r5
                recall_5_count += 1
                query_metrics["recall_at_5"] = round(r5, 4)

                # Recall@10
                r10 = len(ground_truth & set(retrieved_ids[:10])) / len(ground_truth)
                total_recall_10 += r10
                recall_10_count += 1
                query_metrics["recall_at_10"] = round(r10, 4)

                # MRR
                mrr = 0.0
                for rank_idx, cid in enumerate(retrieved_ids, start=1):
                    if cid in ground_truth:
                        mrr = 1.0 / rank_idx
                        break
                total_mrr += mrr
                mrr_count += 1
                query_metrics["mrr"] = round(mrr, 4)

            if expected_source:
                source_hit = expected_source in retrieved_sources
                query_metrics["source_hit"] = source_hit

            results.append(query_metrics)

        # Aggregate metrics
        import numpy as np

        latency_arr = np.array(latencies) if latencies else np.array([0])

        aggregate = {
            "evaluation_id": evaluation_id,
            "total_queries": len(results),
            "params": overrides or {},
            "metrics": {
                "avg_recall_at_5": round(total_recall_5 / recall_5_count, 4) if recall_5_count > 0 else None,
                "avg_recall_at_10": round(total_recall_10 / recall_10_count, 4) if recall_10_count > 0 else None,
                "avg_mrr": round(total_mrr / mrr_count, 4) if mrr_count > 0 else None,
                "avg_latency": round(float(np.mean(latency_arr)), 4),
                "p95_latency": round(float(np.percentile(latency_arr, 95)), 4),
                "min_latency": round(float(np.min(latency_arr)), 4),
                "max_latency": round(float(np.max(latency_arr)), 4),
                "avg_chunks_returned": round(
                    sum(r["chunk_count"] for r in results) / len(results), 2
                ) if results else 0,
            },
            "per_query": results,
        }

        logger.info(
            "Batch evaluation %s: %d queries, avg_recall@5=%s, avg_mrr=%s",
            evaluation_id,
            len(results),
            aggregate["metrics"]["avg_recall_at_5"],
            aggregate["metrics"]["avg_mrr"],
        )

        return aggregate
