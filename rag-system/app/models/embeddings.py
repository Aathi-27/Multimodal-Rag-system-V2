"""
Embeddings - BGE-small-en-v1.5 embedding generation.

Handles:
- Text embedding for chunks (ingestion)
- Query embedding (retrieval)
- Batch embedding with progress
- Dimension: 384, max tokens: 512
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for BGE-small-en-v1.5 embedding generation."""

    DIMENSIONS = 384
    MAX_TOKENS = 512
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_manager) -> None:
        self._manager = model_manager

    @property
    def _model(self):
        return self._manager.get_embedding_model()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of text chunks for indexing.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of shape (len(texts), 384)
        """
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        logger.info("Embedded %d chunks → shape %s", len(texts), embeddings.shape)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query for retrieval.
        Applies the BGE query prefix for asymmetric search.

        Args:
            query: The user's search query.

        Returns:
            numpy array of shape (384,)
        """
        prefixed = self.QUERY_PREFIX + query
        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return embedding

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (no query prefix)."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding
