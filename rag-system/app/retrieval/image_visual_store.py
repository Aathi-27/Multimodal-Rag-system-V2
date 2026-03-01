"""
Image Visual Store — Separate Qdrant collection for CLIP visual embeddings.

Collection: ``image_visual_embeddings``
Vector size: 512 (CLIP ViT-B/32 output)
Distance: Cosine

This collection stores ONE embedding per uploaded image (the raw image
CLIP embedding, NOT the OCR text embedding).  It is searched using the
CLIP text encoder on the query, producing true cross-modal retrieval.

STRICT SEPARATION:
- rag_chunks: 384-dim BGE text embeddings
- image_visual_embeddings: 512-dim CLIP visual embeddings
- Never mix, pad, or normalize between spaces.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class ImageVisualStore:
    """Qdrant collection for CLIP visual embeddings (512-dim, separate space)."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=self._settings.qdrant.host,
                port=self._settings.qdrant.port,
            )
        return self._client

    @property
    def collection_name(self) -> str:
        return self._settings.clip.collection_name

    def ensure_collection(self) -> None:
        """Create the image visual collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info("Image visual collection '%s' exists.", self.collection_name)
            return

        clip_cfg = self._settings.clip
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=clip_cfg.vector_size,
                distance=distance_map.get(clip_cfg.distance, Distance.COSINE),
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info("Created image visual collection: %s (dim=%d)", self.collection_name, clip_cfg.vector_size)

    def upsert_image(
        self,
        image_id: str,
        embedding: np.ndarray,
        payload: dict,
    ) -> None:
        """
        Upsert a single image embedding with metadata.

        Args:
            image_id: Unique identifier (typically upload_id).
            embedding: CLIP visual embedding (512-dim).
            payload: Metadata dict (source, upload_id, file_path, etc.).
        """
        point = PointStruct(
            id=image_id,
            vector=embedding.tolist(),
            payload=payload,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.info("Upserted image visual embedding: %s (%s)", image_id[:8], payload.get("source", "unknown"))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        department_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for visually similar images using CLIP text query embedding.

        Args:
            query_vector: CLIP text embedding (512-dim).
            top_k: Max results.
            department_filter: Optional department filter.

        Returns:
            List of dicts with image_id, score, and payload.
        """
        must_conditions = []
        if department_filter:
            must_conditions.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(value=department_filter),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(hnsw_ef=64),
        )

        return [
            {
                "chunk_id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def count(self) -> int:
        """Get total images in the visual collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def delete_by_upload_id(self, upload_id: str) -> int:
        """Delete visual embedding(s) for a specific upload."""
        try:
            # The point ID IS the upload_id for image uploads
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[upload_id],
            )
            logger.info("Deleted visual embedding for upload %s.", upload_id[:8])
            return 1
        except Exception as e:
            logger.warning("Could not delete visual embedding %s: %s", upload_id[:8], e)
            return 0

    def health_check(self) -> dict:
        """Check image visual collection health."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                return {
                    "status": "not_initialized",
                    "collection": self.collection_name,
                    "image_count": 0,
                }
            count = self.count()
            return {
                "status": "connected",
                "collection": self.collection_name,
                "image_count": count,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
