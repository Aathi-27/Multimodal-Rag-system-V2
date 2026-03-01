"""
Checksum - Index validation to ensure BM25 and Qdrant consistency.

On startup:
1. Count chunks in Qdrant collection
2. Load BM25 index chunk count from metadata
3. Compare: if mismatch → trigger BM25 rebuild from Qdrant

Prevents silent drift between vector and keyword indexes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IndexChecksum:
    """Validates consistency between Qdrant and BM25 indexes."""

    CHECKSUM_FILE = "checksum.json"

    def __init__(self, bm25_path: Path) -> None:
        """
        Args:
            bm25_path: Path to the BM25 index directory (e.g., data/index/current/bm25/).
        """
        self._bm25_path = bm25_path

    @property
    def checksum_file(self) -> Path:
        return self._bm25_path / self.CHECKSUM_FILE

    def save_checksum(self, chunk_count: int, chunk_ids_hash: str) -> None:
        """
        Save the current index checksum after BM25 build/rebuild.

        Args:
            chunk_count: Total number of chunks indexed.
            chunk_ids_hash: SHA256 hash of sorted chunk IDs.
        """
        data = {
            "chunk_count": chunk_count,
            "chunk_ids_hash": chunk_ids_hash,
        }
        self._bm25_path.mkdir(parents=True, exist_ok=True)
        self.checksum_file.write_text(json.dumps(data, indent=2))
        logger.info(
            "Saved BM25 checksum: count=%d, hash=%s",
            chunk_count,
            chunk_ids_hash[:16],
        )

    def load_checksum(self) -> Optional[dict]:
        """Load the stored checksum, or None if not found."""
        if not self.checksum_file.exists():
            return None
        return json.loads(self.checksum_file.read_text())

    @staticmethod
    def compute_ids_hash(chunk_ids: list[str]) -> str:
        """Compute a deterministic hash of chunk IDs for comparison."""
        sorted_ids = sorted(chunk_ids)
        combined = "\n".join(sorted_ids)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def validate(self, qdrant_chunk_count: int, qdrant_chunk_ids: list[str]) -> bool:
        """
        Validate BM25 index against Qdrant state.

        Args:
            qdrant_chunk_count: Number of chunks currently in Qdrant.
            qdrant_chunk_ids: List of all chunk IDs in Qdrant.

        Returns:
            True if consistent, False if rebuild needed.
        """
        stored = self.load_checksum()
        if stored is None:
            logger.warning("No BM25 checksum found. Rebuild required.")
            return False

        if stored["chunk_count"] != qdrant_chunk_count:
            logger.warning(
                "BM25 chunk count mismatch: BM25=%d, Qdrant=%d. Rebuild required.",
                stored["chunk_count"],
                qdrant_chunk_count,
            )
            return False

        current_hash = self.compute_ids_hash(qdrant_chunk_ids)
        if stored["chunk_ids_hash"] != current_hash:
            logger.warning("BM25 chunk IDs hash mismatch. Rebuild required.")
            return False

        logger.info("BM25 index checksum validated successfully.")
        return True
