"""
Vector Store - Qdrant client wrapper.

Handles:
- Collection initialization (384 dim, Cosine, HNSW m=16, ef_construct=100)
- Chunk upsert with metadata
- Vector similarity search with metadata filtering
- Collection statistics
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
    MatchAny,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector database wrapper for chunk storage and retrieval."""

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
        return self._settings.qdrant.collection_name

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info("Qdrant collection '%s' already exists.", self.collection_name)
            return

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self._settings.qdrant.vector_size,
                distance=distance_map.get(
                    self._settings.qdrant.distance, Distance.COSINE
                ),
            ),
            hnsw_config=HnswConfigDiff(
                m=self._settings.qdrant.hnsw_m,
                ef_construct=self._settings.qdrant.hnsw_ef_construct,
            ),
        )
        logger.info("Created Qdrant collection: %s", self.collection_name)

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        payloads: list[dict],
    ) -> None:
        """
        Upsert chunks with embeddings and metadata.

        Args:
            chunk_ids: List of unique chunk IDs.
            embeddings: numpy array of shape (N, 384).
            payloads: List of metadata dicts for each chunk.
        """
        points = [
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload=payload,
            )
            for chunk_id, embedding, payload
            in zip(chunk_ids, embeddings, payloads)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Upserted %d chunks to Qdrant.", len(points))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        modality_filter: Optional[list[str]] = None,
        department_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_vector: Query embedding (384 dim).
            top_k: Number of results to return.
            modality_filter: Filter by modality (e.g., ["document", "audio"]).
            department_filter: Filter by department.

        Returns:
            List of dicts with chunk_id, score, and payload.
        """
        # Build filter conditions
        must_conditions = []
        if modality_filter:
            if len(modality_filter) == 1:
                must_conditions.append(
                    FieldCondition(
                        key="modality",
                        match=MatchValue(value=modality_filter[0]),
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key="modality",
                        match=MatchAny(any=modality_filter),
                    )
                )
        if department_filter:
            must_conditions.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(value=department_filter),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Use query_points (qdrant-client >= 1.12)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(
                hnsw_ef=self._settings.qdrant.hnsw_ef_search,
            ),
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
        """Get total number of chunks in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def get_all_ids(self) -> list[str]:
        """Get all chunk IDs in the collection (for checksum validation)."""
        ids = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            ids.extend(str(p.id) for p in points)
            if next_offset is None:
                break
            offset = next_offset
        return ids

    def health_check(self) -> dict:
        """Check Qdrant connectivity and return status."""
        try:
            collections = self.client.get_collections()
            count = self.count()
            return {
                "status": "connected",
                "collection": self.collection_name,
                "chunk_count": count,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    # ── Knowledge-Base helpers ────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """
        Return aggregated metadata for each unique source document.

        Scrolls through every point and groups by 'source' field.
        Returns a list of dicts, each with:
            source, modality, department, tags, chunk_count, total_tokens, upload_id
        """
        docs: dict[str, dict] = {}
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                payload = p.payload or {}
                src = payload.get("source", "unknown")
                if src not in docs:
                    docs[src] = {
                        "source": src,
                        "modality": payload.get("modality", "document"),
                        "department": payload.get("department", ""),
                        "tags": payload.get("tags", []),
                        "chunk_count": 0,
                        "total_tokens": 0,
                        "upload_id": payload.get("upload_id", ""),
                    }
                docs[src]["chunk_count"] += 1
                docs[src]["total_tokens"] += payload.get("token_count", 0)
            if next_offset is None:
                break
            offset = next_offset
        return list(docs.values())

    def get_chunks_by_source(self, source: str) -> list[dict]:
        """
        Return all chunks belonging to a particular source document.

        Returns list of dicts with: chunk_id, text, chunk_index, page_start, token_count
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks: list[dict] = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                payload = p.payload or {}
                chunks.append({
                    "chunk_id": str(p.id),
                    "text": payload.get("text", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "page_start": payload.get("page_start"),
                    "token_count": payload.get("token_count", 0),
                })
            if next_offset is None:
                break
            offset = next_offset
        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a specific source document.

        Returns the number of chunks deleted.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # First count how many chunks will be removed
        chunks = self.get_chunks_by_source(source)
        if not chunks:
            return 0

        point_ids = [c["chunk_id"] for c in chunks]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )
        logger.info("Deleted %d chunks for source '%s'.", len(point_ids), source)
        return len(point_ids)

    # ── Cross-modal linking helpers ───────────────────────────────────────

    def get_chunks_by_upload_id(
        self,
        upload_id: str,
        with_vectors: bool = False,
    ) -> list[dict]:
        """
        Fetch all chunks belonging to an upload session.

        Args:
            upload_id: The upload session ID to filter by.
            with_vectors: If True, include embedding vectors in results.

        Returns:
            List of dicts with chunk_id, payload, and optionally vector.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks: list[dict] = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                scroll_filter=Filter(
                    must=[FieldCondition(key="upload_id", match=MatchValue(value=upload_id))]
                ),
                with_payload=True,
                with_vectors=with_vectors,
            )
            points, next_offset = result
            for p in points:
                entry: dict = {
                    "chunk_id": str(p.id),
                    "payload": p.payload or {},
                }
                if with_vectors and p.vector is not None:
                    entry["vector"] = p.vector
                chunks.append(entry)
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
        with_vectors: bool = False,
    ) -> list[dict]:
        """
        Fetch specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve.
            with_vectors: If True, include embedding vectors.

        Returns:
            List of dicts with chunk_id and payload (only found points).
        """
        if not chunk_ids:
            return []

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=with_vectors,
        )

        results = []
        for p in points:
            entry: dict = {
                "chunk_id": str(p.id),
                "payload": p.payload or {},
            }
            if with_vectors and p.vector is not None:
                entry["vector"] = p.vector
            results.append(entry)
        return results

    def update_payloads_batch(self, updates: list[dict]) -> None:
        """
        Update payload fields on existing points.

        Args:
            updates: List of dicts, each with:
                - chunk_id: Point ID to update
                - payload_update: Dict of fields to set/overwrite
        """
        for update in updates:
            chunk_id = update["chunk_id"]
            payload_update = update["payload_update"]
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload_update,
                points=[chunk_id],
            )
        logger.info("Updated payloads for %d chunks.", len(updates))
