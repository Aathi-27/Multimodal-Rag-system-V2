"""
CLIP Encoder — Visual + Text embedding via openai/clip-vit-base-patch32.

Provides:
- Image embedding (512-dim) from file paths or PIL Images
- Text embedding (512-dim) for CLIP text space
- Lazy model loading via ModelManager

The CLIP text space is DIFFERENT from BGE text space (384-dim).
These embeddings live in a separate Qdrant collection.
Do NOT mix, merge, or normalize between the two spaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Wrapper for CLIP ViT-B/32 image and text encoding."""

    DIMENSIONS = 512

    def __init__(self, model_manager) -> None:
        self._manager = model_manager
        self._settings = get_settings()
        self._available: Optional[bool] = None

    @property
    def _model(self):
        return self._manager.get_clip_model()

    @property
    def is_available(self) -> bool:
        """Check if CLIP is enabled and model can be loaded."""
        if self._available is not None:
            return self._available
        if not self._settings.clip.enabled:
            self._available = False
            return False
        try:
            _ = self._model
            self._available = True
        except Exception as e:
            logger.warning("CLIP model not available: %s", e)
            self._available = False
        return self._available

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Encode a single image file to a 512-dim CLIP embedding.

        Args:
            image_path: Path to the image file (PNG, JPEG, WEBP).

        Returns:
            numpy array of shape (512,) or None if encoding fails.
        """
        if not self._settings.clip.enabled:
            return None

        try:
            from PIL import Image

            path = Path(image_path)
            if not path.exists():
                logger.error("Image not found: %s", image_path)
                return None

            img = Image.open(path).convert("RGB")
            embedding = self._model.encode(
                img,
                normalize_embeddings=True,
            )
            logger.debug("CLIP image encoded: %s → shape %s", path.name, embedding.shape)
            return embedding

        except Exception as e:
            logger.error("CLIP image encoding failed for %s: %s", image_path, e)
            return None

    def encode_images_batch(
        self,
        image_paths: list[str],
        batch_size: int = 8,
    ) -> list[Optional[np.ndarray]]:
        """
        Encode multiple images to CLIP embeddings.

        Returns list of embeddings (None for failed images).
        """
        if not self._settings.clip.enabled:
            return [None] * len(image_paths)

        from PIL import Image

        results: list[Optional[np.ndarray]] = []
        images_to_encode: list = []
        valid_indices: list[int] = []

        for i, path_str in enumerate(image_paths):
            try:
                path = Path(path_str)
                if path.exists():
                    img = Image.open(path).convert("RGB")
                    images_to_encode.append(img)
                    valid_indices.append(i)
                else:
                    logger.warning("Image not found: %s", path_str)
            except Exception as e:
                logger.warning("Could not open image %s: %s", path_str, e)

        # Initialize results with None
        results = [None] * len(image_paths)

        if images_to_encode:
            try:
                embeddings = self._model.encode(
                    images_to_encode,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(images_to_encode) > 10,
                )
                for idx, emb in zip(valid_indices, embeddings):
                    results[idx] = emb
            except Exception as e:
                logger.error("CLIP batch encoding failed: %s", e)

        return results

    def encode_text(self, query: str) -> Optional[np.ndarray]:
        """
        Encode a text query into CLIP text space (512-dim).

        This is NOT the same space as BGE text embeddings.
        Used for searching the image_visual_embeddings collection.

        Args:
            query: User's text query.

        Returns:
            numpy array of shape (512,) or None if encoding fails.
        """
        if not self._settings.clip.enabled:
            return None

        try:
            embedding = self._model.encode(
                query,
                normalize_embeddings=True,
            )
            return embedding
        except Exception as e:
            logger.error("CLIP text encoding failed: %s", e)
            return None
