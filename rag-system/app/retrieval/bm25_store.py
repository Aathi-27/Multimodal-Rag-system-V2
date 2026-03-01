"""
BM25 Store - Keyword search index manager.

Handles:
- Building BM25 index from chunk texts
- Serializing index to pickle file
- Loading index from disk
- Checksum validation against Qdrant on startup
- Rebuilding from Qdrant if checksum mismatch
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.processing.normalization import normalize_text
from app.versioning.checksum import IndexChecksum

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 keyword search index backed by rank-bm25."""

    BM25_FILE = "bm25_index.pkl"
    METADATA_FILE = "bm25_metadata.pkl"

    def __init__(self, index_dir: Path) -> None:
        """
        Args:
            index_dir: Path to the BM25 index directory (e.g., data/index/current/bm25/).
        """
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._chunk_metadata: list[dict] = []
        self._checksum = IndexChecksum(index_dir)

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    def build(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        chunk_metadata: list[dict],
    ) -> None:
        """
        Build a BM25 index from chunk texts.

        Args:
            chunk_ids: List of unique chunk IDs (aligned with Qdrant).
            chunk_texts: List of chunk text strings.
            chunk_metadata: List of metadata dicts per chunk.
        """
        if not chunk_texts:
            logger.warning("No chunks to index. BM25 index will be empty.")
            return

        # Tokenize (simple whitespace for BM25)
        tokenized = [text.lower().split() for text in chunk_texts]

        self._bm25 = BM25Okapi(tokenized)
        self._chunk_ids = chunk_ids
        self._chunk_metadata = chunk_metadata

        logger.info("BM25 index built with %d chunks.", len(chunk_ids))

    def save(self) -> None:
        """Serialize BM25 index and metadata to disk."""
        if self._bm25 is None:
            raise RuntimeError("No BM25 index to save. Call build() first.")

        bm25_path = self._index_dir / self.BM25_FILE
        meta_path = self._index_dir / self.METADATA_FILE

        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "metadata": self._chunk_metadata},
                f,
            )

        # Save checksum
        ids_hash = IndexChecksum.compute_ids_hash(self._chunk_ids)
        self._checksum.save_checksum(len(self._chunk_ids), ids_hash)

        logger.info("BM25 index saved to %s", bm25_path)

    def load(self) -> bool:
        """
        Load BM25 index from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        bm25_path = self._index_dir / self.BM25_FILE
        meta_path = self._index_dir / self.METADATA_FILE

        if not bm25_path.exists() or not meta_path.exists():
            logger.warning("BM25 index files not found at %s", self._index_dir)
            return False

        with open(bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)

        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._chunk_metadata = data["metadata"]

        logger.info("BM25 index loaded: %d chunks.", len(self._chunk_ids))
        return True

    def validate_against_qdrant(
        self, qdrant_count: int, qdrant_ids: list[str]
    ) -> bool:
        """
        Validate BM25 index consistency against Qdrant.

        Returns:
            True if consistent, False if rebuild needed.
        """
        return self._checksum.validate(qdrant_count, qdrant_ids)

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """
        Search the BM25 index with a query.

        Args:
            query: User's search query (will be normalized).
            top_k: Number of top results to return.

        Returns:
            List of dicts with chunk_id, score, and rank.
        """
        if not self.is_loaded:
            raise RuntimeError("BM25 index not loaded. Call load() or build() first.")

        # Normalize and tokenize query
        normalized = normalize_text(query)
        query_tokens = normalized.lower().split()

        if not query_tokens:
            return []

        # Get scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self._chunk_ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                    "metadata": self._chunk_metadata[idx] if idx < len(self._chunk_metadata) else {},
                })

        return results
