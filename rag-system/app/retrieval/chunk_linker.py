"""
Chunk Linker — Payload-based adjacency linking for cross-modal chunk expansion.

After ingestion, computes pairwise cosine similarity between session chunks
and writes ``related_chunk_ids`` back into Qdrant payloads.  At retrieval
time, the hybrid retriever expands top-k results using these links.

Linking strategies
------------------
1. **Embedding similarity** — cosine sim above adaptive threshold
2. **Timestamp overlap** — audio chunks with overlapping time ranges
3. **Source bundle grouping** — chunks from co-uploaded files (same bundle_id)

Constraints
-----------
- ``max_related_per_chunk``: 3 (configurable, never exceed)
- Similarity threshold is **NOT** hardcoded — derived from distribution
  analysis, with adaptive fallback when configured value is too aggressive.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class ChunkLinker:
    """Computes and persists inter-chunk links based on embedding similarity."""

    def __init__(self, vector_store) -> None:
        from app.retrieval.vector_store import VectorStore

        self._vector: VectorStore = vector_store
        self._settings = get_settings()

    # ── Public API ────────────────────────────────────────────────────────

    def link_session(self, upload_id: str) -> dict:
        """
        Compute and persist links for all chunks in an upload session.

        Steps:
            1. Fetch all chunks + vectors for the upload_id
            2. Compute pairwise cosine similarity
            3. Log similarity distribution (for threshold calibration)
            4. Assign ``related_chunk_ids`` (top-N above threshold)
            5. Check timestamp overlaps for audio chunks
            6. Write links back to Qdrant payloads

        Returns:
            Summary dict with distribution stats and link counts.
        """
        linking_cfg = self._settings.linking

        if not linking_cfg.enabled:
            logger.info("[linker] Linking disabled, skipping session %s", upload_id[:8])
            return {"status": "disabled"}

        # Fetch all chunks with vectors for this session
        chunks_with_vectors = self._vector.get_chunks_by_upload_id(
            upload_id=upload_id,
            with_vectors=True,
        )

        n_chunks = len(chunks_with_vectors)
        if n_chunks < 2:
            logger.info(
                "[linker] Session %s has %d chunk(s), skipping linking",
                upload_id[:8], n_chunks,
            )
            return {"status": "skipped", "reason": "too_few_chunks", "count": n_chunks}

        chunk_ids = [c["chunk_id"] for c in chunks_with_vectors]
        vectors = np.array([c["vector"] for c in chunks_with_vectors])
        payloads = [c["payload"] for c in chunks_with_vectors]

        # ── Pairwise cosine similarity ────────────────────────────────────
        sim_matrix = self._cosine_similarity_matrix(vectors)

        # ── Distribution analysis ─────────────────────────────────────────
        distribution = self._analyze_distribution(sim_matrix)
        logger.info(
            "[linker] Session %s similarity distribution: "
            "mean=%.4f, std=%.4f, p50=%.4f, p75=%.4f, p90=%.4f, p95=%.4f",
            upload_id[:8],
            distribution["mean"], distribution["std"],
            distribution["p50"], distribution["p75"],
            distribution["p90"], distribution["p95"],
        )

        # ── Adaptive threshold ────────────────────────────────────────────
        threshold = self._resolve_threshold(
            configured=linking_cfg.similarity_threshold,
            distribution=distribution,
        )

        # ── Build link map ────────────────────────────────────────────────
        max_per_chunk = linking_cfg.max_related_per_chunk
        link_map: dict[str, list[str]] = {}

        for i, cid in enumerate(chunk_ids):
            candidates: list[tuple[str, float]] = []
            for j, other_id in enumerate(chunk_ids):
                if i == j:
                    continue
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    candidates.append((other_id, score))

            # Sort by similarity descending, take top-N
            candidates.sort(key=lambda x: x[1], reverse=True)
            link_map[cid] = [c[0] for c in candidates[:max_per_chunk]]

        # ── Timestamp overlap links (audio chunks) ────────────────────────
        ts_links = self._find_timestamp_overlaps(
            chunk_ids, payloads, linking_cfg.timestamp_overlap_seconds,
        )
        for cid, extra_ids in ts_links.items():
            existing = set(link_map.get(cid, []))
            for linked_id in extra_ids:
                if linked_id not in existing and len(link_map.get(cid, [])) < max_per_chunk:
                    link_map.setdefault(cid, []).append(linked_id)
                    existing.add(linked_id)

        # ── Source bundle links ───────────────────────────────────────────
        bundle_links = self._find_source_bundle_links(chunk_ids, payloads)
        for cid, extra_ids in bundle_links.items():
            existing = set(link_map.get(cid, []))
            for linked_id in extra_ids:
                if linked_id not in existing and len(link_map.get(cid, [])) < max_per_chunk:
                    link_map.setdefault(cid, []).append(linked_id)
                    existing.add(linked_id)

        # ── Write links back to Qdrant ────────────────────────────────────
        total_links = 0
        updates: list[dict] = []
        for cid, related_ids in link_map.items():
            if related_ids:
                updates.append({
                    "chunk_id": cid,
                    "payload_update": {"related_chunk_ids": related_ids},
                })
                total_links += len(related_ids)

        if updates:
            self._vector.update_payloads_batch(updates)

        summary = {
            "status": "completed",
            "session_chunk_count": n_chunks,
            "chunks_with_links": sum(1 for v in link_map.values() if v),
            "total_links_created": total_links,
            "effective_threshold": round(threshold, 4),
            "distribution": distribution,
        }

        logger.info(
            "[linker] Session %s linked: %d/%d chunks, %d links (threshold=%.4f)",
            upload_id[:8],
            summary["chunks_with_links"],
            n_chunks,
            total_links,
            threshold,
        )

        return summary

    def analyze_session_distribution(self, upload_id: str) -> dict:
        """
        Analyze similarity distribution for a session WITHOUT creating links.

        Used for threshold calibration before committing.
        """
        chunks_with_vectors = self._vector.get_chunks_by_upload_id(
            upload_id=upload_id,
            with_vectors=True,
        )

        if len(chunks_with_vectors) < 2:
            return {"status": "insufficient_chunks", "count": len(chunks_with_vectors)}

        vectors = np.array([c["vector"] for c in chunks_with_vectors])
        sim_matrix = self._cosine_similarity_matrix(vectors)
        distribution = self._analyze_distribution(sim_matrix)

        return {
            "status": "ok",
            "chunk_count": len(chunks_with_vectors),
            "distribution": distribution,
            "recommended_threshold": round(distribution["p75"], 4),
        }

    def analyze_global_distribution(self, sample_limit: int = 500) -> dict:
        """
        Analyze similarity distribution across ALL chunks (sampled).

        Useful for establishing a global baseline threshold.
        """
        # Scroll up to sample_limit chunks with vectors
        from qdrant_client.models import Filter

        points: list = []
        offset = None
        while len(points) < sample_limit:
            batch, next_offset = self._vector.client.scroll(
                collection_name=self._vector.collection_name,
                limit=min(500, sample_limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if len(points) < 2:
            return {"status": "insufficient_chunks", "count": len(points)}

        vectors = np.array([p.vector for p in points])
        sim_matrix = self._cosine_similarity_matrix(vectors)
        distribution = self._analyze_distribution(sim_matrix)

        return {
            "status": "ok",
            "chunk_count": len(points),
            "distribution": distribution,
            "recommended_threshold": round(distribution["p75"], 4),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_threshold(configured: float, distribution: dict) -> float:
        """
        Resolve the effective similarity threshold.

        If the configured threshold exceeds p90, it is likely too aggressive
        for the embedding space.  Fall back to p75, floored at 0.50.
        """
        p90 = distribution.get("p90", 0.0)
        p75 = distribution.get("p75", 0.0)

        if configured > p90 > 0:
            adaptive = max(p75, 0.50)
            logger.warning(
                "[linker] Configured threshold %.4f > p90 (%.4f). "
                "Falling back to p75-based adaptive threshold: %.4f",
                configured, p90, adaptive,
            )
            return adaptive

        return configured

    @staticmethod
    def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix (N×N)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        normalized = vectors / norms
        return normalized @ normalized.T

    @staticmethod
    def _analyze_distribution(sim_matrix: np.ndarray) -> dict:
        """
        Extract upper-triangle values (excluding diagonal) and compute
        descriptive statistics for threshold calibration.
        """
        n = sim_matrix.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        values = sim_matrix[upper_idx]

        if len(values) == 0:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0,
                "pair_count": 0,
            }

        return {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "pair_count": int(len(values)),
        }

    @staticmethod
    def _find_timestamp_overlaps(
        chunk_ids: list[str],
        payloads: list[dict],
        overlap_threshold_seconds: float,
    ) -> dict[str, list[str]]:
        """
        Find audio chunks with overlapping timestamp ranges.

        Only applies to chunks where modality == "audio" and both
        ``timestamp_start`` and ``timestamp_end`` are present.
        """
        audio_chunks: list[dict] = []
        for cid, payload in zip(chunk_ids, payloads):
            if payload.get("modality") != "audio":
                continue
            ts_start = payload.get("timestamp_start")
            ts_end = payload.get("timestamp_end")
            if ts_start is not None and ts_end is not None:
                try:
                    audio_chunks.append({
                        "chunk_id": cid,
                        "start": float(ts_start),
                        "end": float(ts_end),
                    })
                except (ValueError, TypeError):
                    continue

        overlaps: dict[str, list[str]] = {}
        for i, a in enumerate(audio_chunks):
            for j, b in enumerate(audio_chunks):
                if i >= j:
                    continue
                overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
                if overlap >= overlap_threshold_seconds:
                    overlaps.setdefault(a["chunk_id"], []).append(b["chunk_id"])
                    overlaps.setdefault(b["chunk_id"], []).append(a["chunk_id"])

        return overlaps

    @staticmethod
    def _find_source_bundle_links(
        chunk_ids: list[str],
        payloads: list[dict],
    ) -> dict[str, list[str]]:
        """
        Group chunks by ``source_bundle_id`` and create cross-source links.

        Only links chunks that have different ``source`` values but share
        the same non-empty ``source_bundle_id``.
        """
        # Group by bundle
        bundles: dict[str, list[tuple[str, str]]] = {}
        for cid, payload in zip(chunk_ids, payloads):
            bundle_id = payload.get("source_bundle_id", "")
            if not bundle_id:
                continue
            source = payload.get("source", "")
            bundles.setdefault(bundle_id, []).append((cid, source))

        links: dict[str, list[str]] = {}
        for bundle_id, members in bundles.items():
            if len(members) < 2:
                continue
            # Link chunks from different sources within the same bundle
            for i, (cid_a, src_a) in enumerate(members):
                for j, (cid_b, src_b) in enumerate(members):
                    if i >= j or src_a == src_b:
                        continue
                    links.setdefault(cid_a, []).append(cid_b)
                    links.setdefault(cid_b, []).append(cid_a)

        return links
