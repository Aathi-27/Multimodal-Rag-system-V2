"""
Model Registry - Central metadata registry for all ML models.

Tracks model paths, expected checksums, and load status for:
- Embedding (BGE-small-en-v1.5)
- Reranker (bge-reranker-base)
- LLM (Qwen2.5-1.5B-Instruct GGUF)
- SLM (optional small model for query rewrite / compression)
- Whisper (faster-whisper-small)
- OCR (EasyOCR)
- Vision (MiniCPM-V, optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    LLM = "llm"
    SLM = "slm"
    WHISPER = "whisper"
    OCR = "ocr"
    VISION = "vision"
    CLIP_VISUAL = "clip_visual"


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    model_type: ModelType
    name: str
    path: Path
    description: str
    required: bool = True
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    extra: dict = field(default_factory=dict)


class ModelRegistry:
    """Central registry that tracks all model metadata and availability."""

    def __init__(self, models_root: Path) -> None:
        self._models_root = models_root
        self._registry: dict[ModelType, ModelInfo] = {}
        self._register_defaults()
        self.check_availability()          # set DOWNLOADED for models on disk

    def _register_defaults(self) -> None:
        """Register all expected models with their default paths."""
        root = self._models_root

        self._registry[ModelType.EMBEDDING] = ModelInfo(
            model_type=ModelType.EMBEDDING,
            name="BAAI/bge-small-en-v1.5",
            path=root / "embeddings" / "bge-small-en-v1.5",
            description="384-dim embedding model for semantic search",
            extra={"dimensions": 384, "max_tokens": 512},
        )
        self._registry[ModelType.RERANKER] = ModelInfo(
            model_type=ModelType.RERANKER,
            name="BAAI/bge-reranker-base",
            path=root / "reranker" / "bge-reranker-base",
            description="Cross-encoder reranker for result refinement",
        )
        self._registry[ModelType.LLM] = ModelInfo(
            model_type=ModelType.LLM,
            name="Qwen2.5-1.5B-Instruct-Q4_K_M",
            path=root / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            description="Qwen2.5 1.5B quantized GGUF for generation",
            extra={"context_window": 32768, "quantization": "Q4_K_M"},
        )
        self._registry[ModelType.SLM] = ModelInfo(
            model_type=ModelType.SLM,
            name="SLM (optional)",
            path=root / "llm" / "slm.gguf",
            description="Optional small model for query rewrite / compression",
            required=False,
        )
        self._registry[ModelType.WHISPER] = ModelInfo(
            model_type=ModelType.WHISPER,
            name="faster-whisper-small",
            path=root / "whisper" / "faster-whisper-small",
            description="CTranslate2 Whisper for speech-to-text",
        )
        self._registry[ModelType.OCR] = ModelInfo(
            model_type=ModelType.OCR,
            name="EasyOCR",
            path=root / "ocr",
            description="EasyOCR for image text extraction (auto-downloads models)",
        )
        self._registry[ModelType.VISION] = ModelInfo(
            model_type=ModelType.VISION,
            name="MiniCPM-V-2.6",
            path=root / "vision" / "minicpm-v",
            description="Vision-language model for image captioning",
            required=False,
        )
        self._registry[ModelType.CLIP_VISUAL] = ModelInfo(
            model_type=ModelType.CLIP_VISUAL,
            name="openai/clip-vit-base-patch32",
            path=root / "clip" / "clip-ViT-B-32",
            description="CLIP ViT-B/32 for visual semantic search (512-dim)",
            required=False,
            extra={"dimensions": 512, "model_name": "clip-ViT-B-32"},
        )

    def get(self, model_type: ModelType) -> ModelInfo:
        """Get model info by type."""
        if model_type not in self._registry:
            raise KeyError(f"Model type {model_type} not registered.")
        return self._registry[model_type]

    def check_availability(self) -> dict[ModelType, bool]:
        """Check which models are downloaded and available on disk."""
        results: dict[ModelType, bool] = {}
        for model_type, info in self._registry.items():
            exists = info.path.exists()
            if exists:
                info.status = ModelStatus.DOWNLOADED
            else:
                info.status = ModelStatus.NOT_DOWNLOADED
            results[model_type] = exists
        return results

    def get_missing_required(self) -> list[ModelInfo]:
        """Return list of required models that are not downloaded."""
        self.check_availability()
        return [
            info
            for info in self._registry.values()
            if info.required and info.status == ModelStatus.NOT_DOWNLOADED
        ]

    def all_models(self) -> list[ModelInfo]:
        """Return all registered models."""
        return list(self._registry.values())
