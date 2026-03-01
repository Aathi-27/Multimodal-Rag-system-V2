"""
Failure Diagnosis — Automated root-cause classification for retrieval failures.

Analyzes queries that returned poor results and classifies the likely cause:
  - corpus_gap         — The corpus simply doesn't contain relevant content
  - embedding_mismatch — Query/document embedding similarity is too low
  - rerank_threshold   — Good candidates were dropped by the reranker threshold
  - rrf_dilution       — Too many irrelevant chunks diluted the fusion stage
  - entity_miss        — Entity injection failed to find relevant chunks
  - parameter_issue    — Current settings are suboptimal for this query type

Each diagnosis includes confidence, evidence, and actionable recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Diagnosis:
    """Root-cause diagnosis for a query."""
    root_cause: str
    confidence: float  # 0.0 - 1.0
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    secondary_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "secondary_causes": self.secondary_causes,
        }


class FailureDiagnoser:
    """
    Automated failure diagnosis engine.

    Analyzes a query record (including survival log and recall validation)
    to determine why the retrieval may have failed.
    """

    # Root cause labels
    CORPUS_GAP = "corpus_gap"
    EMBEDDING_MISMATCH = "embedding_mismatch"
    RERANK_THRESHOLD = "rerank_threshold"
    RRF_DILUTION = "rrf_dilution"
    ENTITY_MISS = "entity_miss"
    PARAMETER_ISSUE = "parameter_issue"
    NO_FAILURE = "no_failure"

    def diagnose(self, query_record: dict) -> Diagnosis:
        """
        Analyze a query record and classify the root cause of poor retrieval.

        Args:
            query_record: Full query record from QueryStore (includes
                         survival_log, recall_validation, debug_info, etc.)

        Returns:
            Diagnosis with root_cause, confidence, evidence, and recommendations.
        """
        signals = self._gather_signals(query_record)
        causes = self._score_causes(signals)

        # Select the top cause
        if not causes:
            return Diagnosis(
                root_cause=self.NO_FAILURE,
                confidence=0.5,
                evidence=["Insufficient data to diagnose"],
                recommendations=["Enable debug mode and run recall validation"],
            )

        causes.sort(key=lambda c: c[1], reverse=True)
        top_cause, top_score, top_evidence, top_recs = causes[0]

        secondary = [c[0] for c in causes[1:3] if c[1] > 0.3]

        return Diagnosis(
            root_cause=top_cause,
            confidence=min(top_score, 1.0),
            evidence=top_evidence,
            recommendations=top_recs,
            secondary_causes=secondary,
        )

    def diagnose_batch(self, records: list[dict]) -> dict:
        """
        Diagnose multiple queries and return aggregate statistics.

        Returns:
            cause_distribution  — { root_cause: count }
            avg_confidence      — Mean confidence across diagnoses
            common_recs         — Most frequent recommendations
            diagnoses           — List of individual diagnosis dicts
        """
        diagnoses = [self.diagnose(r) for r in records]

        cause_dist: dict[str, int] = {}
        rec_counter: dict[str, int] = {}
        total_confidence = 0.0

        for d in diagnoses:
            cause_dist[d.root_cause] = cause_dist.get(d.root_cause, 0) + 1
            total_confidence += d.confidence
            for rec in d.recommendations:
                rec_counter[rec] = rec_counter.get(rec, 0) + 1

        avg_conf = total_confidence / len(diagnoses) if diagnoses else 0.0
        common_recs = sorted(rec_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_analyzed": len(records),
            "cause_distribution": cause_dist,
            "avg_confidence": round(avg_conf, 3),
            "common_recommendations": [{"recommendation": r, "frequency": c} for r, c in common_recs],
            "diagnoses": [d.to_dict() for d in diagnoses],
        }

    def _gather_signals(self, record: dict) -> dict:
        """Extract diagnostic signals from a query record."""
        signals: dict = {}

        # Basic query info
        signals["has_answer"] = bool(record.get("answer"))
        signals["chunk_count"] = record.get("chunk_count", 0)
        signals["has_error"] = bool(record.get("error"))

        # Survival log analysis
        survival_log = record.get("survival_log", [])
        if survival_log:
            signals["total_chunks_seen"] = len(survival_log)
            signals["survived_count"] = sum(1 for e in survival_log if e.get("survived"))
            signals["dropped_count"] = signals["total_chunks_seen"] - signals["survived_count"]

            # Count by drop stage
            drop_stages: dict[str, int] = {}
            for entry in survival_log:
                if entry.get("dropped_at"):
                    stage = entry["dropped_at"]
                    drop_stages[stage] = drop_stages.get(stage, 0) + 1
            signals["drop_stages"] = drop_stages

            # Vector vs BM25 presence
            vector_present = sum(
                1 for e in survival_log if e.get("stages", {}).get("vector", {}).get("present")
            )
            bm25_present = sum(
                1 for e in survival_log if e.get("stages", {}).get("bm25", {}).get("present")
            )
            signals["vector_present"] = vector_present
            signals["bm25_present"] = bm25_present

            # Reranker drop analysis
            reranker_drops = sum(
                1 for e in survival_log
                if e.get("dropped_at") == "reranker"
            )
            signals["reranker_drops"] = reranker_drops

            # Check if high-scored vector results were dropped
            vector_scores = [
                e["stages"]["vector"].get("score", 0)
                for e in survival_log
                if e.get("stages", {}).get("vector", {}).get("present")
                and not e.get("survived")
            ]
            signals["dropped_vector_scores"] = sorted(vector_scores, reverse=True)[:5]
        else:
            signals["total_chunks_seen"] = 0
            signals["survived_count"] = 0

        # Recall validation analysis
        recall = record.get("recall_validation", {})
        if recall:
            signals["has_recall"] = True
            signals["recall_at_5"] = recall.get("recall_at_5")
            signals["recall_at_10"] = recall.get("recall_at_10")
            signals["mrr"] = recall.get("mrr")
            signals["relevant_count"] = recall.get("relevant_count", 0)
            signals["annotated_count"] = recall.get("annotated_count", 0)
        else:
            signals["has_recall"] = False

        # Debug info analysis
        debug = record.get("debug_info", {})
        if debug:
            signals["has_debug"] = True
            signals["effective_params"] = debug.get("effective_params", {})
            signals["entity_info"] = debug.get("entity_info", {})
        else:
            signals["has_debug"] = False

        return signals

    def _score_causes(self, signals: dict) -> list[tuple]:
        """
        Score each possible root cause based on signals.

        Returns list of (cause, score, evidence, recommendations).
        """
        causes = []

        # ── 1. Corpus Gap ─────────────────────────────────────────────
        score = 0.0
        evidence = []
        recs = []

        if signals.get("total_chunks_seen", 0) == 0:
            score += 0.8
            evidence.append("No chunks appeared in any pipeline stage")
        elif signals.get("vector_present", 0) < 3:
            score += 0.5
            evidence.append(f"Only {signals.get('vector_present', 0)} vector matches found")
        if signals.get("bm25_present", 0) < 3:
            score += 0.2
            evidence.append(f"Only {signals.get('bm25_present', 0)} BM25 matches found")
        if signals.get("chunk_count", 0) == 0:
            score += 0.3
            evidence.append("Zero chunks in final results")

        if score > 0.3:
            recs.append("Upload more documents related to this topic")
            recs.append("Check if the corpus covers the query domain")
            causes.append((self.CORPUS_GAP, score, evidence, recs))

        # ── 2. Rerank Threshold Issue ─────────────────────────────────
        score = 0.0
        evidence = []
        recs = []

        reranker_drops = signals.get("reranker_drops", 0)
        if reranker_drops > 3:
            score += 0.6
            evidence.append(f"{reranker_drops} chunks dropped by reranker threshold")
        elif reranker_drops > 0:
            score += 0.3
            evidence.append(f"{reranker_drops} chunks dropped by reranker")

        if signals.get("has_recall") and signals.get("recall_at_5") is not None:
            r5 = signals["recall_at_5"]
            r10 = signals.get("recall_at_10", r5)
            if r10 is not None and r5 is not None and r10 > r5:
                score += 0.2
                evidence.append(f"Recall improves from @5 ({r5}) to @10 ({r10})")

        if score > 0.2:
            recs.append("Lower rerank_threshold (current may be too aggressive)")
            recs.append("Increase rerank_min_results to retain more candidates")
            causes.append((self.RERANK_THRESHOLD, score, evidence, recs))

        # ── 3. Embedding Mismatch ─────────────────────────────────────
        score = 0.0
        evidence = []
        recs = []

        dropped_vector_scores = signals.get("dropped_vector_scores", [])
        if dropped_vector_scores and max(dropped_vector_scores) < 0.3:
            score += 0.5
            evidence.append(f"Highest dropped vector score: {dropped_vector_scores[0]:.4f}")
        if signals.get("vector_present", 0) > 0 and signals.get("bm25_present", 0) > signals.get("vector_present", 0) * 2:
            score += 0.3
            evidence.append("BM25 finds significantly more results than vector search")
            evidence.append("Possible embedding model vocabulary mismatch")

        if score > 0.3:
            recs.append("Consider a larger embedding model for this domain")
            recs.append("Review if query terminology matches document language")
            causes.append((self.EMBEDDING_MISMATCH, score, evidence, recs))

        # ── 4. RRF Dilution ───────────────────────────────────────────
        score = 0.0
        evidence = []
        recs = []

        drop_stages = signals.get("drop_stages", {})
        rrf_drops = drop_stages.get("rerank_selection", 0)
        if rrf_drops > 5:
            score += 0.4
            evidence.append(f"{rrf_drops} chunks dropped at rerank selection (too many fused)")
        if signals.get("total_chunks_seen", 0) > 30 and signals.get("survived_count", 0) < 3:
            score += 0.3
            evidence.append("Many chunks seen but very few survived")

        if score > 0.3:
            recs.append("Reduce vector_top_k and bm25_top_k to reduce noise")
            recs.append("Increase rerank_count to evaluate more candidates")
            causes.append((self.RRF_DILUTION, score, evidence, recs))

        # ── 5. Parameter Issue ────────────────────────────────────────
        score = 0.0
        evidence = []
        recs = []

        params = signals.get("effective_params", {})
        if params:
            if params.get("rerank_count", 15) < 10:
                score += 0.3
                evidence.append(f"rerank_count={params.get('rerank_count')} is low")
                recs.append("Increase rerank_count to at least 15")
            if params.get("rerank_threshold", 0.15) > 0.3:
                score += 0.3
                evidence.append(f"rerank_threshold={params.get('rerank_threshold')} is high")
                recs.append("Lower rerank_threshold closer to 0.15")

        if score > 0.2:
            causes.append((self.PARAMETER_ISSUE, score, evidence, recs))

        return causes
