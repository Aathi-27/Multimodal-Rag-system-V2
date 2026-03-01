"""Image Query Retriever - Strategy A (CLIP image -> linked expansion -> rerank)
                         + OCR fallback (image -> OCR text -> hybrid retrieval -> LLM).

Primary pipeline (CLIP):
  1. Encode query image -> CLIP image vector (512-d)
  2. Search image_visual_embeddings collection
  3. Expand via related_chunk_ids to text chunks
  4. RRF-merge image hits + linked text hits
  5. Rerank text chunks only (reranker requires text, not images)
  6. Return fused results with origin tracking

OCR fallback (fires when CLIP yields 0 results):
  1. Run EasyOCR on the query image -> extract text
  2. Combine OCR text + user prompt as a text query
  3. Search knowledge base via hybrid retriever (vector + BM25)
  4. Return text results with ocr_text attached

Fusion rules (Phase 4.3):
  - Image-origin penalty: x0.95
  - Linked-origin penalty: x0.9
  - Cap max_image_results = clip.max_image_results (default 5)
  - Direct text relevance dominates

STRICT: CLIP vectors NEVER search the BGE text collection.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class ImageQueryRetriever:
    """Retrieves content using an image as the query (Strategy A + OCR fallback)."""

    def __init__(
        self,
        image_visual_store,
        clip_encoder,
        vector_store,
        reranker,
        hybrid_retriever=None,
        embedding_model=None,
    ) -> None:
        self._image_visual = image_visual_store
        self._clip_encoder = clip_encoder
        self._vector = vector_store
        self._reranker = reranker
        self._hybrid = hybrid_retriever
        self._embedding_model = embedding_model
        self._settings = get_settings()

    def retrieve(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        top_k: int = 10,
        department_filter: Optional[str] = None,
        debug: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Full image-as-query retrieval pipeline.

        Args:
            image_path: Path to the query image file.
            text_prompt: Optional text prompt to accompany the image.
            top_k: Max final results.
            department_filter: Optional department filter.
            debug: If True, return (results, debug_info) tuple.

        Returns:
            List of ranked results. If debug=True, (results, debug_info).
        """
        clip_cfg = self._settings.clip
        linking_cfg = self._settings.linking
        timings: dict[str, float] = {}

        # ── Stage 1: CLIP image encoding ─────────────────────────────
        t0 = time.perf_counter()
        image_embedding = self._clip_encoder.encode_image(image_path)
        timings["image_embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if timings["image_embedding_ms"] > clip_cfg.image_embedding_timeout_ms:
            logger.warning(
                "Image embedding exceeded guardrail: %.1fms > %dms limit",
                timings["image_embedding_ms"], clip_cfg.image_embedding_timeout_ms,
            )

        if image_embedding is None:
            logger.error("CLIP image encoding failed for: %s", image_path)
            if debug:
                return [], {"error": "CLIP image encoding failed", "timings": timings}
            return []

        # ── Stage 2: Search image_visual_embeddings ──────────────────
        t0 = time.perf_counter()
        image_results = self._image_visual.search(
            query_vector=image_embedding,
            top_k=clip_cfg.max_image_results,
            department_filter=department_filter,
        )
        timings["image_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if timings["image_search_ms"] > clip_cfg.image_search_timeout_ms:
            logger.warning(
                "Image search exceeded guardrail: %.1fms > %dms limit",
                timings["image_search_ms"], clip_cfg.image_search_timeout_ms,
            )

        logger.info(
            "Image query: %d visual matches (search=%.1fms)",
            len(image_results), timings["image_search_ms"],
        )

        # ── Stage 3: Linked expansion to text chunks ─────────────────
        # Image results have upload_ids → find their text chunks → get related_chunk_ids
        t0 = time.perf_counter()
        linked_text_chunks: list[dict] = []
        image_entries: list[dict] = []

        for rank, ir in enumerate(image_results, start=1):
            img_upload_id = ir.get("payload", {}).get("upload_id", ir.get("chunk_id", ""))
            penalized_score = ir["score"] * clip_cfg.modality_penalty

            image_entries.append({
                "chunk_id": ir["chunk_id"],
                "rrf_score": penalized_score,
                "rrf_rank": rank,
                "metadata": ir.get("payload", {}),
                "origin": "image_query",
                "clip_score": round(ir["score"], 4),
            })

            # Find text chunks from the same upload session
            if linking_cfg.enabled:
                try:
                    session_chunks = self._vector.get_chunks_by_upload_id(
                        upload_id=img_upload_id,
                        with_vectors=False,
                    )

                    for sc in session_chunks:
                        # Also expand via related_chunk_ids
                        related_ids = sc.get("payload", {}).get("related_chunk_ids", [])
                        for rid in related_ids[:linking_cfg.max_related_per_chunk]:
                            linked_text_chunks.append({
                                "chunk_id": rid,
                                "parent_image_id": ir["chunk_id"],
                                "parent_score": penalized_score,
                            })

                        # Include the text chunk itself (it's from the same image upload)
                        linked_text_chunks.append({
                            "chunk_id": sc["chunk_id"],
                            "parent_image_id": ir["chunk_id"],
                            "parent_score": penalized_score,
                        })
                except Exception as e:
                    logger.warning("Linked expansion failed for upload %s: %s", img_upload_id[:8], e)

        timings["linked_expansion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Deduplicate linked chunks
        seen_ids: set[str] = {e["chunk_id"] for e in image_entries}
        unique_linked: list[dict] = []
        for lc in linked_text_chunks:
            if lc["chunk_id"] not in seen_ids:
                seen_ids.add(lc["chunk_id"])
                unique_linked.append(lc)
        unique_linked = unique_linked[:linking_cfg.max_total_expansion]

        # Fetch actual chunk data from Qdrant text collection
        linked_entries: list[dict] = []
        if unique_linked:
            fetch_ids = [lc["chunk_id"] for lc in unique_linked]
            fetched = self._vector.get_chunks_by_ids(fetch_ids)
            fetched_map = {c["chunk_id"]: c for c in fetched}

            for lc in unique_linked:
                chunk_data = fetched_map.get(lc["chunk_id"])
                if chunk_data:
                    linked_score = lc["parent_score"] * linking_cfg.expansion_penalty
                    linked_entries.append({
                        "chunk_id": lc["chunk_id"],
                        "rrf_score": linked_score,
                        "rrf_rank": len(image_entries) + len(linked_entries) + 1,
                        "metadata": chunk_data.get("payload", {}),
                        "origin": "image_linked",
                        "linked_from_image": lc["parent_image_id"],
                        "clip_score": None,
                    })

        logger.info(
            "Image linked expansion: %d text chunks from %d images",
            len(linked_entries), len(image_entries),
        )

        # ── Stage 4: Optional text prompt search ─────────────────────
        # If user provides a text prompt, also search CLIP text space
        text_search_entries: list[dict] = []
        if text_prompt and text_prompt.strip():
            t0 = time.perf_counter()
            clip_text_emb = self._clip_encoder.encode_text(text_prompt)
            if clip_text_emb is not None:
                text_visual_results = self._image_visual.search(
                    query_vector=clip_text_emb,
                    top_k=clip_cfg.max_image_results,
                    department_filter=department_filter,
                )
                for tr in text_visual_results:
                    if tr["chunk_id"] not in seen_ids:
                        seen_ids.add(tr["chunk_id"])
                        text_search_entries.append({
                            "chunk_id": tr["chunk_id"],
                            "rrf_score": tr["score"] * clip_cfg.modality_penalty * 0.8,  # Extra penalty for text-assisted
                            "rrf_rank": len(image_entries) + len(linked_entries) + len(text_search_entries) + 1,
                            "metadata": tr.get("payload", {}),
                            "origin": "text_assisted_image",
                            "clip_score": round(tr["score"], 4),
                        })
            timings["text_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 5: Merge all candidates ────────────────────────────
        all_candidates = image_entries + linked_entries + text_search_entries
        all_candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

        # ── Stage 6: Rerank text chunks only ─────────────────────────
        # Reranker requires text. Image entries have no text to rerank.
        t0 = time.perf_counter()
        text_candidates = [c for c in all_candidates if c.get("origin") in ("image_linked",)]
        image_only = [c for c in all_candidates if c.get("origin") not in ("image_linked",)]

        reranked_text: list[dict] = []
        if text_candidates and text_prompt:
            rerank_input = text_candidates[:self._settings.retrieval.rerank_count]
            reranked_text = self._reranker.rerank(
                query=text_prompt,
                candidates=rerank_input,
                threshold=self._settings.retrieval.rerank_threshold,
                min_results=self._settings.retrieval.rerank_min_results,
            )
        elif text_candidates:
            # No text prompt → skip reranking, keep order
            reranked_text = text_candidates
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Merge: image hits first, then reranked text
        final_results = image_only + reranked_text
        final_results = final_results[:top_k]

        # Re-number ranks
        for i, r in enumerate(final_results, start=1):
            r["rrf_rank"] = i

        timings["total_ms"] = round(sum(v for v in timings.values() if isinstance(v, (int, float))), 2)

        # ── Stage 7: OCR fallback (when CLIP yields nothing) ─────────
        ocr_text: str = ""
        ocr_results: list[dict] = []
        used_ocr_fallback = False

        if not final_results and self._hybrid and self._embedding_model:
            t0 = time.perf_counter()
            ocr_text = self._ocr_extract(image_path)
            timings["ocr_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            if ocr_text.strip():
                # Build a combined query from OCR text + user prompt
                combined_query = ocr_text
                if text_prompt and text_prompt.strip():
                    combined_query = f"{text_prompt}\n\nImage text: {ocr_text}"

                t0 = time.perf_counter()
                query_emb = self._embedding_model.embed_query(combined_query)
                ocr_output = self._hybrid.retrieve(
                    query=combined_query,
                    query_embedding=query_emb,
                    top_k=top_k,
                    department_filter=department_filter,
                    debug=debug,
                )
                timings["ocr_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                if debug and isinstance(ocr_output, tuple):
                    ocr_results, _ = ocr_output
                else:
                    ocr_results = ocr_output if isinstance(ocr_output, list) else []

                # Tag OCR results
                for r in ocr_results:
                    r["origin"] = f"ocr_fallback:{r.get('origin', 'unknown')}"

                final_results = ocr_results
                used_ocr_fallback = True
                logger.info(
                    "OCR fallback: extracted %d chars → %d results in %.1fms",
                    len(ocr_text), len(ocr_results), timings.get("ocr_retrieval_ms", 0),
                )
            else:
                logger.info("OCR fallback: no text extracted from image")

        # Re-number ranks on final results
        for i, r in enumerate(final_results, start=1):
            r["rrf_rank"] = i

        timings["total_ms"] = round(sum(v for v in timings.values() if isinstance(v, (int, float))), 2)

        logger.info(
            "Image query complete: %d results (images=%d, linked=%d, text_assist=%d, ocr=%d) in %.1fms",
            len(final_results), len(image_only), len(reranked_text),
            len(text_search_entries), len(ocr_results), timings["total_ms"],
        )

        if debug:
            debug_info = {
                "retrieval_mode": "image",
                "image_results": image_entries,
                "linked_results": linked_entries,
                "text_search_results": text_search_entries,
                "ocr_fallback": used_ocr_fallback,
                "ocr_text": ocr_text if used_ocr_fallback else None,
                "final_results": final_results,
                "timings": timings,
                "modality_contribution": {
                    "image_hits": len(image_entries),
                    "linked_hits": len(linked_entries),
                    "text_assist_hits": len(text_search_entries),
                    "ocr_fallback_hits": len(ocr_results),
                },
            }
            return final_results, debug_info

        return final_results, {
            "ocr_fallback": used_ocr_fallback,
            "ocr_text": ocr_text if used_ocr_fallback else None,
            "timings": timings,
        }

    @staticmethod
    def _ocr_extract(image_path: str) -> str:
        """Extract text from an image using EasyOCR (lightweight, no GPU)."""
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            results = reader.readtext(image_path)
            texts = [text for _, text, conf in results if conf >= 0.5]
            return " ".join(texts)
        except Exception as e:
            logger.warning("OCR extraction failed: %s", e)
            return ""
