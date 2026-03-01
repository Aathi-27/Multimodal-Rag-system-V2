"""
Hybrid Retriever - Vector + BM25 + Reciprocal Rank Fusion.

Three-stage retrieval:
1. Parallel vector search (Qdrant) and BM25 keyword search
2. Reciprocal Rank Fusion (RRF) with k=60
3. Cross-encoder reranking (top 15)

RRF formula: score = Σ [1 / (k + rank)]
- k=60 (default constant)
- Deduplication by chunk_id before fusion
- Preserves highest rank when chunk appears in multiple sources
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config.settings import get_settings
from app.retrieval.bm25_store import BM25Store
from app.retrieval.entity_extractor import extract_entities, find_entity_chunks
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector search and BM25 with RRF fusion and reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: Reranker,
        image_visual_store=None,
        clip_encoder=None,
    ) -> None:
        self._vector = vector_store
        self._bm25 = bm25_store
        self._reranker = reranker
        self._image_visual = image_visual_store
        self._clip_encoder = clip_encoder
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        modality_filter: Optional[list[str]] = None,
        department_filter: Optional[str] = None,
        debug: bool = False,
        overrides: Optional[dict] = None,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Full hybrid retrieval pipeline.

        Steps:
        1. Vector search (top 50)
        2. BM25 search (top 50)
        3. RRF fusion (k=60)
        4. Cross-encoder reranking (top 15)

        Args:
            query: User's query text.
            query_embedding: Pre-computed query embedding (384 dim).
            top_k: Max final results (after reranking).
            modality_filter: Filter by content modality.
            department_filter: Filter by department.
            debug: If True, return (results, debug_info) tuple.
            overrides: Optional runtime parameter overrides
                       (rrf_k, rerank_threshold, rerank_count, rerank_min_results).

        Returns:
            List of ranked results with scores and metadata.
            If debug=True, returns (results, debug_info) tuple.
        """
        settings = self._settings.retrieval
        ovr = overrides or {}

        # Resolve effective parameters (overrides take precedence)
        vector_top_k = ovr.get("vector_top_k", settings.vector_top_k)
        bm25_top_k = ovr.get("bm25_top_k", settings.bm25_top_k)
        rrf_k = ovr.get("rrf_k", settings.rrf_k)
        rerank_count = ovr.get("rerank_count", settings.rerank_count)
        rerank_threshold = ovr.get("rerank_threshold", settings.rerank_threshold)
        rerank_min_results = ovr.get("rerank_min_results", settings.rerank_min_results)

        # ── Stage 1: Parallel search ─────────────────────────────────────
        vector_results = self._vector.search(
            query_vector=query_embedding,
            top_k=vector_top_k,
            modality_filter=modality_filter,
            department_filter=department_filter,
        )

        bm25_results = self._bm25.search(
            query=query,
            top_k=bm25_top_k,
        )

        logger.info(
            "Search results: vector=%d, bm25=%d",
            len(vector_results),
            len(bm25_results),
        )

        # ── Stage 2: RRF Fusion ──────────────────────────────────────────
        fused = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=rrf_k,
        )

        logger.info("RRF fusion: %d unique candidates", len(fused))

        # ── Stage 2b: Entity injection ───────────────────────────────────
        entities = extract_entities(query)
        entity_chunks: list[dict] = []
        if entities:
            memory_cfg = self._settings.memory
            entity_chunks = find_entity_chunks(
                entities=entities,
                bm25_store=self._bm25,
                per_entity_limit=memory_cfg.entity_per_limit,
                global_limit=memory_cfg.entity_global_limit,
            )
            # Merge entity chunks into fused results (avoid duplicates)
            fused_ids = {r["chunk_id"] for r in fused}
            for ec in entity_chunks:
                if ec["chunk_id"] not in fused_ids:
                    fused.append({
                        "chunk_id": ec["chunk_id"],
                        "rrf_score": 0.0,  # Entity-injected, no RRF score
                        "rrf_rank": len(fused) + 1,
                        "metadata": ec.get("payload", ec.get("metadata", {})),
                        "origin": "entity",
                        "vector_rank": None,
                        "bm25_rank": None,
                        "entity_match": ec.get("entity_match", ""),
                    })
                    fused_ids.add(ec["chunk_id"])

        # ── Stage 2c: Linked chunk expansion ─────────────────────────────
        linking_cfg = self._settings.linking
        linked_chunks: list[dict] = []

        if linking_cfg.enabled:
            fused_ids = {r["chunk_id"] for r in fused}

            # Collect expansion candidates from related_chunk_ids
            expansion_candidates: list[dict] = []
            for r in fused:
                meta = r.get("metadata", r.get("payload", {}))
                related_ids = meta.get("related_chunk_ids", [])
                parent_score = r.get("rrf_score", 0.0)

                for rid in related_ids[:linking_cfg.max_related_per_chunk]:
                    if rid not in fused_ids:
                        expansion_candidates.append({
                            "chunk_id": rid,
                            "parent_score": parent_score,
                            "parent_chunk_id": r["chunk_id"],
                        })

            # Deduplicate and cap
            seen: set[str] = set()
            unique_expansions: list[dict] = []
            for ec in expansion_candidates:
                if ec["chunk_id"] not in seen and ec["chunk_id"] not in fused_ids:
                    seen.add(ec["chunk_id"])
                    unique_expansions.append(ec)
            unique_expansions = unique_expansions[:linking_cfg.max_total_expansion]

            # Fetch chunk data from Qdrant
            if unique_expansions:
                expansion_ids = [ec["chunk_id"] for ec in unique_expansions]
                fetched = self._vector.get_chunks_by_ids(expansion_ids)
                fetched_map = {c["chunk_id"]: c for c in fetched}

                for ec in unique_expansions:
                    chunk_data = fetched_map.get(ec["chunk_id"])
                    if chunk_data:
                        expanded_score = ec["parent_score"] * linking_cfg.expansion_penalty
                        entry = {
                            "chunk_id": ec["chunk_id"],
                            "rrf_score": expanded_score,
                            "rrf_rank": len(fused) + len(linked_chunks) + 1,
                            "metadata": chunk_data.get("payload", {}),
                            "origin": "linked",
                            "vector_rank": None,
                            "bm25_rank": None,
                            "linked_from": ec["parent_chunk_id"],
                        }
                        linked_chunks.append(entry)
                        fused_ids.add(ec["chunk_id"])

                fused.extend(linked_chunks)
                # Re-sort by rrf_score so reranking sees the best candidates
                fused.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

                logger.info(
                    "Linked expansion: %d chunks injected (penalty=%.2f)",
                    len(linked_chunks), linking_cfg.expansion_penalty,
                )

        # ── Stage 2d: Image visual branch (CLIP) ────────────────────────
        clip_cfg = self._settings.clip
        image_results: list[dict] = []
        visual_intent: dict = {"has_visual_intent": False, "confidence": 0.0, "matched_keywords": []}

        if (
            clip_cfg.enabled
            and self._image_visual is not None
            and self._clip_encoder is not None
        ):
            from app.retrieval.visual_intent_detector import detect_visual_intent

            visual_intent = detect_visual_intent(query)

            if visual_intent["has_visual_intent"]:
                try:
                    clip_query_emb = self._clip_encoder.encode_text(query)
                    if clip_query_emb is not None:
                        raw_image_results = self._image_visual.search(
                            query_vector=clip_query_emb,
                            top_k=clip_cfg.max_image_results,
                            department_filter=department_filter,
                        )

                        fused_ids = {r["chunk_id"] for r in fused}
                        for ir in raw_image_results:
                            if ir["chunk_id"] not in fused_ids:
                                # Apply modality penalty to CLIP score
                                penalized_score = ir["score"] * clip_cfg.modality_penalty
                                # Map CLIP score to RRF-scale for fair comparison
                                # CLIP scores are 0-1 cosine; scale to approximate RRF range
                                rrf_equiv = penalized_score * (1.0 / (rrf_k + 1))
                                entry = {
                                    "chunk_id": ir["chunk_id"],
                                    "rrf_score": rrf_equiv,
                                    "rrf_rank": len(fused) + len(image_results) + 1,
                                    "metadata": ir.get("payload", {}),
                                    "origin": "image_visual",
                                    "vector_rank": None,
                                    "bm25_rank": None,
                                    "clip_score": round(ir["score"], 4),
                                }
                                image_results.append(entry)
                                fused_ids.add(ir["chunk_id"])

                        if image_results:
                            fused.extend(image_results)
                            fused.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
                            logger.info(
                                "Image branch: %d results injected (penalty=%.2f, intent=%.1f)",
                                len(image_results),
                                clip_cfg.modality_penalty,
                                visual_intent["confidence"],
                            )
                except Exception as e:
                    logger.warning("Image branch failed (non-fatal): %s", e)

        # Build origin lookup for propagation to reranked / dropped
        origin_map = {r["chunk_id"]: r.get("origin", "unknown") for r in fused}
        vector_rank_map = {r["chunk_id"]: r.get("vector_rank") for r in fused}
        bm25_rank_map = {r["chunk_id"]: r.get("bm25_rank") for r in fused}

        effective_params = {
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "rrf_k": rrf_k,
            "rerank_count": rerank_count,
            "rerank_threshold": rerank_threshold,
            "rerank_min_results": rerank_min_results,
        }

        entity_debug = {
            "extracted_entities": entities,
            "injected_count": len(entity_chunks),
            "injected_chunks": [
                {
                    "chunk_id": ec["chunk_id"],
                    "entity_match": ec.get("entity_match", ""),
                    "source": ec.get("payload", ec.get("metadata", {})).get("source", "unknown"),
                }
                for ec in entity_chunks
            ],
        }

        linking_debug = {
            "enabled": linking_cfg.enabled,
            "injected_count": len(linked_chunks),
            "expansion_penalty": linking_cfg.expansion_penalty,
            "injected_chunks": [
                {
                    "chunk_id": lc["chunk_id"],
                    "linked_from": lc.get("linked_from", ""),
                    "score": round(lc.get("rrf_score", 0.0), 4),
                    "source": lc.get("metadata", {}).get("source", "unknown"),
                }
                for lc in linked_chunks
            ],
        }

        image_branch_debug = {
            "enabled": clip_cfg.enabled,
            "visual_intent": visual_intent,
            "injected_count": len(image_results),
            "modality_penalty": clip_cfg.modality_penalty,
            "injected_images": [
                {
                    "chunk_id": ir["chunk_id"],
                    "clip_score": ir.get("clip_score", 0.0),
                    "source": ir.get("metadata", {}).get("source", "unknown"),
                }
                for ir in image_results
            ],
        }

        if not fused:
            if debug:
                return [], {
                    "vector_results": vector_results,
                    "bm25_results": bm25_results,
                    "rrf_fused": [],
                    "reranked": [],
                    "dropped": [],
                    "effective_params": effective_params,
                    "entity_info": entity_debug,
                    "linking_info": linking_debug,
                    "image_branch_info": image_branch_debug,
                }
            return []

        # ── Stage 3: Cross-encoder reranking ─────────────────────────────
        rerank_candidates = fused[:rerank_count]
        reranked = self._reranker.rerank(
            query=query,
            candidates=rerank_candidates,
            threshold=rerank_threshold,
            min_results=rerank_min_results,
        )

        # Propagate origin / source-ranks into reranked results
        for r in reranked:
            cid = r["chunk_id"]
            r.setdefault("origin", origin_map.get(cid, "unknown"))
            r.setdefault("vector_rank", vector_rank_map.get(cid))
            r.setdefault("bm25_rank", bm25_rank_map.get(cid))

        logger.info("Reranked: %d results", len(reranked))

        if debug:
            # Identify dropped candidates (below threshold, not in final)
            reranked_ids = {r["chunk_id"] for r in reranked}
            dropped = [c for c in rerank_candidates if c["chunk_id"] not in reranked_ids]
            for d in dropped:
                cid = d["chunk_id"]
                d.setdefault("origin", origin_map.get(cid, "unknown"))
                d.setdefault("vector_rank", vector_rank_map.get(cid))
                d.setdefault("bm25_rank", bm25_rank_map.get(cid))

            # ── Compute survival log ─────────────────────────────────
            from app.observability.survival_tracker import compute_survival_log

            survival_log = compute_survival_log(
                vector_results=vector_results,
                bm25_results=bm25_results,
                rrf_fused=fused,
                rerank_candidates=rerank_candidates,
                reranked=reranked,
                entity_chunks=entity_chunks,
                linked_chunks=linked_chunks,
                image_results=image_results,
                effective_params=effective_params,
            )

            debug_info = {
                "vector_results": vector_results[:20],
                "bm25_results": bm25_results[:20],
                "rrf_fused": fused[:20],
                "reranked": reranked,
                "dropped": dropped,
                "effective_params": effective_params,
                "entity_info": entity_debug,
                "linking_info": linking_debug,
                "image_branch_info": image_branch_debug,
                "survival_log": survival_log,
            }
            return reranked, debug_info

        return reranked

    @staticmethod
    def _rrf_fusion(
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion with per-chunk origin tracking.

        RRF score = Σ [1 / (k + rank)]

        Rules:
        - Deduplicate by chunk_id before fusion
        - Preserve highest rank when chunk appears in both sources
        - Track origin: "vector", "bm25", or "both"
        - k=60 (default constant)
        """
        scores: dict[str, float] = {}
        metadata_map: dict[str, dict] = {}
        origins: dict[str, set[str]] = {}
        vector_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        # Score vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            origins.setdefault(chunk_id, set()).add("vector")
            vector_ranks[chunk_id] = rank
            if chunk_id not in metadata_map:
                metadata_map[chunk_id] = result.get("payload", result.get("metadata", {}))

        # Score BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            origins.setdefault(chunk_id, set()).add("bm25")
            bm25_ranks[chunk_id] = rank
            if chunk_id not in metadata_map:
                metadata_map[chunk_id] = result.get("payload", result.get("metadata", {}))

        # Sort by fused RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        fused_results = []
        for rank, chunk_id in enumerate(sorted_ids, start=1):
            origin_set = origins.get(chunk_id, set())
            origin_label = "both" if len(origin_set) > 1 else next(iter(origin_set))
            fused_results.append({
                "chunk_id": chunk_id,
                "rrf_score": scores[chunk_id],
                "rrf_rank": rank,
                "metadata": metadata_map.get(chunk_id, {}),
                "origin": origin_label,
                "vector_rank": vector_ranks.get(chunk_id),
                "bm25_rank": bm25_ranks.get(chunk_id),
            })

        return fused_results
