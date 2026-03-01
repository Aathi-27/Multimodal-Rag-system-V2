"""
Model Manager - Lazy loading and lifecycle management for all ML models.

Ensures models are loaded on-demand and properly released.
Tracks memory usage and provides health status.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.models.model_registry import ModelRegistry, ModelStatus, ModelType

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lazy loading, caching, and unloading of ML models."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._loaded_models: dict[ModelType, Any] = {}

    def get_embedding_model(self):
        """Get or load the BGE embedding model."""
        if ModelType.EMBEDDING not in self._loaded_models:
            self._load_embedding()
        return self._loaded_models[ModelType.EMBEDDING]

    def get_reranker_model(self):
        """Get or load the BGE reranker model."""
        if ModelType.RERANKER not in self._loaded_models:
            self._load_reranker()
        return self._loaded_models[ModelType.RERANKER]

    def get_clip_model(self):
        """Get or load the CLIP visual model."""
        if ModelType.CLIP_VISUAL not in self._loaded_models:
            self._load_clip()
        return self._loaded_models[ModelType.CLIP_VISUAL]

    def _load_embedding(self) -> None:
        """Load BGE-small-en-v1.5 via sentence-transformers."""
        info = self._registry.get(ModelType.EMBEDDING)
        logger.info("Loading embedding model: %s", info.name)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(info.path),
            device="cuda",  # GPU for faster query encoding
        )
        self._loaded_models[ModelType.EMBEDDING] = model
        info.status = ModelStatus.LOADED
        logger.info("Embedding model loaded successfully.")

    def _load_reranker(self) -> None:
        """Load BGE-reranker-base via sentence-transformers CrossEncoder."""
        info = self._registry.get(ModelType.RERANKER)
        logger.info("Loading reranker model: %s", info.name)

        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            str(info.path),
            device="cpu",
        )
        self._loaded_models[ModelType.RERANKER] = model
        info.status = ModelStatus.LOADED
        logger.info("Reranker model loaded successfully.")

    def _load_clip(self) -> None:
        """Load CLIP ViT-B/32 for visual embedding.

        Auto-downloads from HuggingFace if not cached locally.
        Uses sentence-transformers which handles both image and text encoding.
        """
        from app.config.settings import get_settings

        clip_cfg = get_settings().clip
        info = self._registry.get(ModelType.CLIP_VISUAL)
        logger.info("Loading CLIP model: %s (device=%s)", info.name, clip_cfg.device)

        from sentence_transformers import SentenceTransformer

        # Try local path first, fall back to model name for auto-download
        model_path = str(info.path) if info.path.exists() else clip_cfg.model_name
        model = SentenceTransformer(
            model_path,
            device=clip_cfg.device,
        )

        # If loaded from remote, save locally for future offline use
        if not info.path.exists():
            try:
                info.path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(info.path))
                logger.info("CLIP model saved to %s for offline use.", info.path)
            except Exception as e:
                logger.warning("Could not save CLIP model locally: %s", e)

        self._loaded_models[ModelType.CLIP_VISUAL] = model
        info.status = ModelStatus.LOADED
        logger.info("CLIP model loaded successfully (dim=%d).", clip_cfg.vector_size)

    def unload(self, model_type: ModelType) -> None:
        """Unload a model from memory."""
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
            info = self._registry.get(model_type)
            info.status = ModelStatus.DOWNLOADED
            logger.info("Model unloaded: %s", info.name)

    def unload_all(self) -> None:
        """Unload all models from memory."""
        for model_type in list(self._loaded_models.keys()):
            self.unload(model_type)

    def health_status(self) -> dict[str, str]:
        """Return load status for all registered models."""
        return {
            mt.value: self._registry.get(mt).status.value
            for mt in ModelType
        }
