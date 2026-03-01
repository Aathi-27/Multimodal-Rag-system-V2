"""
Survival Tracker — Stage-wise chunk survival logging.

Tracks every chunk's journey through the retrieval pipeline:
  Vector Search → BM25 Search → RRF Fusion → Entity Injection → Reranking → Final

For each chunk, records which stages it appeared in, its rank/score at each
stage, and if dropped, the reason why.  This enables:
  - Visualizing where good chunks are lost
  - Identifying threshold/parameter issues
  - Comparing stage-level behavior across queries
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_survival_log(
    vector_results: list[dict],
    bm25_results: list[dict],
    rrf_fused: list[dict],
    rerank_candidates: list[dict],
    reranked: list[dict],
    entity_chunks: list[dict] | None = None,
    linked_chunks: list[dict] | None = None,
    image_results: list[dict] | None = None,
    image_query_results: list[dict] | None = None,
    effective_params: dict | None = None,
) -> list[dict]:
    """
    Build a per-chunk survival log across all pipeline stages.

    Each entry in the returned list represents one unique chunk that
    appeared in ANY stage.  Fields:

        chunk_id        — Unique chunk identifier
        source          — Document source name
        stages          — Dict of stage → presence info:
            vector:       { present, rank, score }
            bm25:         { present, rank, score }
            rrf:          { present, rank, score, origin }
            entity:       { present, entity_match }
            linked:       { present, linked_from, score }
            image_branch: { present, clip_score }
            reranker:     { present, rank, score }
            final:        { present, rank }
        survived        — True if chunk is in the final results
        dropped_at      — Stage name where the chunk was eliminated (or null)
        dropped_reason  — Human-readable reason for elimination
    """
    params = effective_params or {}

    # ── Index all chunks by ID across stages ──────────────────────────
    all_chunk_ids: list[str] = []
    chunk_meta: dict[str, dict] = {}  # chunk_id → metadata

    def _cid(item: dict) -> str:
        return str(item.get("chunk_id", ""))

    def _meta(item: dict) -> dict:
        m = item.get("metadata", item.get("payload", {}))
        return {"source": m.get("source", "unknown"), "text_preview": m.get("text", "")[:80]}

    def _register(cid: str, item: dict) -> None:
        if cid and cid not in chunk_meta:
            all_chunk_ids.append(cid)
            chunk_meta[cid] = _meta(item)

    # Register all chunks
    for r in vector_results:
        _register(_cid(r), r)
    for r in bm25_results:
        _register(_cid(r), r)
    for r in rrf_fused:
        _register(_cid(r), r)
    for r in (entity_chunks or []):
        _register(_cid(r), r)
    for r in (linked_chunks or []):
        _register(_cid(r), r)
    for r in (image_results or []):
        _register(_cid(r), r)
    for r in (image_query_results or []):
        _register(_cid(r), r)
    for r in rerank_candidates:
        _register(_cid(r), r)
    for r in reranked:
        _register(_cid(r), r)

    # ── Build lookup maps for each stage ──────────────────────────────
    vector_map: dict[str, dict] = {}
    for i, r in enumerate(vector_results):
        cid = _cid(r)
        if cid:
            vector_map[cid] = {
                "rank": i + 1,
                "score": round(float(r.get("score", 0)), 4),
            }

    bm25_map: dict[str, dict] = {}
    for i, r in enumerate(bm25_results):
        cid = _cid(r)
        if cid:
            bm25_map[cid] = {
                "rank": r.get("rank", i + 1),
                "score": round(float(r.get("score", 0)), 4),
            }

    rrf_map: dict[str, dict] = {}
    for i, r in enumerate(rrf_fused):
        cid = _cid(r)
        if cid:
            rrf_map[cid] = {
                "rank": i + 1,
                "score": round(float(r.get("rrf_score", 0)), 4),
                "origin": r.get("origin", "unknown"),
                "vector_rank": r.get("vector_rank"),
                "bm25_rank": r.get("bm25_rank"),
            }

    entity_map: dict[str, dict] = {}
    for r in (entity_chunks or []):
        cid = _cid(r)
        if cid:
            entity_map[cid] = {
                "entity_match": r.get("entity_match", ""),
            }

    linked_map: dict[str, dict] = {}
    for r in (linked_chunks or []):
        cid = _cid(r)
        if cid:
            linked_map[cid] = {
                "linked_from": r.get("linked_from", ""),
                "score": round(float(r.get("rrf_score", 0)), 4),
            }

    image_map: dict[str, dict] = {}
    for r in (image_results or []):
        cid = _cid(r)
        if cid:
            image_map[cid] = {
                "clip_score": round(float(r.get("clip_score", r.get("score", 0))), 4),
            }

    image_query_map: dict[str, dict] = {}
    for r in (image_query_results or []):
        cid = _cid(r)
        if cid:
            image_query_map[cid] = {
                "clip_score": round(float(r.get("clip_score", r.get("score", 0))), 4),
                "origin": r.get("origin", "image_query"),
            }

    rerank_candidate_set = {_cid(r) for r in rerank_candidates}

    reranked_map: dict[str, dict] = {}
    for i, r in enumerate(reranked):
        cid = _cid(r)
        if cid:
            reranked_map[cid] = {
                "rank": i + 1,
                "score": round(float(r.get("reranker_score", 0)), 4),
            }

    final_set = set(reranked_map.keys())
    rerank_count = params.get("rerank_count", 15)
    rerank_threshold = params.get("rerank_threshold", 0.15)

    # ── Build survival log ────────────────────────────────────────────
    survival_log: list[dict] = []

    for cid in all_chunk_ids:
        meta = chunk_meta.get(cid, {})

        stages = {
            "vector": {
                "present": cid in vector_map,
                **(vector_map[cid] if cid in vector_map else {}),
            },
            "bm25": {
                "present": cid in bm25_map,
                **(bm25_map[cid] if cid in bm25_map else {}),
            },
            "rrf": {
                "present": cid in rrf_map,
                **(rrf_map[cid] if cid in rrf_map else {}),
            },
            "entity": {
                "present": cid in entity_map,
                **(entity_map[cid] if cid in entity_map else {}),
            },
            "linked": {
                "present": cid in linked_map,
                **(linked_map[cid] if cid in linked_map else {}),
            },
            "image_branch": {
                "present": cid in image_map,
                **(image_map[cid] if cid in image_map else {}),
            },
            "image_query_branch": {
                "present": cid in image_query_map,
                **(image_query_map[cid] if cid in image_query_map else {}),
            },
            "reranker": {
                "present": cid in reranked_map,
                **(reranked_map[cid] if cid in reranked_map else {}),
            },
            "final": {
                "present": cid in final_set,
                "rank": reranked_map[cid]["rank"] if cid in reranked_map else None,
            },
        }

        # Determine where the chunk was dropped
        survived = cid in final_set
        dropped_at: Optional[str] = None
        dropped_reason: Optional[str] = None

        if not survived:
            if cid not in rrf_map and cid not in entity_map and cid not in linked_map and cid not in image_map and cid not in image_query_map:
                # Never made it past individual search to fusion
                if cid in vector_map and cid not in bm25_map:
                    dropped_at = "rrf_fusion"
                    dropped_reason = "Only in vector results, not fused"
                elif cid in bm25_map and cid not in vector_map:
                    dropped_at = "rrf_fusion"
                    dropped_reason = "Only in BM25 results, not fused"
                else:
                    dropped_at = "rrf_fusion"
                    dropped_reason = "Low RRF score, below fusion cutoff"
            elif cid in rrf_map and cid not in rerank_candidate_set:
                dropped_at = "rerank_selection"
                dropped_reason = f"RRF rank too low (not in top {rerank_count})"
            elif cid in rerank_candidate_set and cid not in reranked_map:
                dropped_at = "reranker"
                dropped_reason = f"Reranker score below threshold ({rerank_threshold})"
            elif cid in entity_map and cid not in rerank_candidate_set:
                dropped_at = "rerank_selection"
                dropped_reason = "Entity-injected but not selected for reranking"
            elif cid in linked_map and cid not in rerank_candidate_set:
                dropped_at = "rerank_selection"
                dropped_reason = "Link-expanded but not selected for reranking"
            elif cid in image_map and cid not in rerank_candidate_set:
                dropped_at = "rerank_selection"
                dropped_reason = "Image-visual result but not selected for reranking"
            elif cid in image_query_map and cid not in rerank_candidate_set:
                dropped_at = "rerank_selection"
                dropped_reason = "Image-query result but not selected for reranking"
            else:
                dropped_at = "unknown"
                dropped_reason = "Eliminated at unknown stage"

        entry = {
            "chunk_id": cid,
            "source": meta.get("source", "unknown"),
            "stages": stages,
            "survived": survived,
            "dropped_at": dropped_at,
            "dropped_reason": dropped_reason,
        }
        survival_log.append(entry)

    logger.info(
        "Survival log: %d total chunks, %d survived, %d dropped",
        len(survival_log),
        sum(1 for e in survival_log if e["survived"]),
        sum(1 for e in survival_log if not e["survived"]),
    )

    return survival_log


def compute_survival_summary(survival_log: list[dict]) -> dict:
    """
    Compute aggregate survival statistics from a survival log.

    Returns:
        total_chunks_seen       — Unique chunks across all stages
        survived_count          — Chunks in final results
        dropped_count           — Chunks eliminated
        drop_reasons            — Counter of drop reasons
        stage_counts            — { stage_name: count_of_chunks_present }
        survival_rates          — { stage_name: fraction_of_total }
    """
    total = len(survival_log)
    survived = sum(1 for e in survival_log if e["survived"])
    dropped = total - survived

    # Count drop reasons
    drop_reasons: dict[str, int] = {}
    for e in survival_log:
        if e["dropped_at"]:
            reason = e["dropped_at"]
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    # Stage presence counts
    stage_counts: dict[str, int] = {}
    for stage_name in ("vector", "bm25", "rrf", "entity", "linked", "image_branch", "image_query_branch", "reranker", "final"):
        stage_counts[stage_name] = sum(
            1 for e in survival_log if e["stages"].get(stage_name, {}).get("present", False)
        )

    survival_rates = {
        stage: round(count / total, 4) if total > 0 else 0.0
        for stage, count in stage_counts.items()
    }

    return {
        "total_chunks_seen": total,
        "survived_count": survived,
        "dropped_count": dropped,
        "drop_reasons": drop_reasons,
        "stage_counts": stage_counts,
        "survival_rates": survival_rates,
    }
