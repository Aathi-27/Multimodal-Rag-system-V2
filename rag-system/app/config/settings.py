"""
Settings - Configuration loader for the RAG system.

Loads from config.yaml with environment variable overrides.
All system parameters are centralized here.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Project root: rag-system/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class ChunkingConfig:
    target_tokens: int = 480
    max_tokens: int = 512
    overlap_tokens: int = 50
    method: str = "sliding_window"


@dataclass
class RetrievalConfig:
    vector_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    rerank_count: int = 10
    rerank_threshold: float = 0.15
    rerank_min_results: int = 5


@dataclass
class MemoryConfig:
    entity_per_limit: int = 10
    entity_global_limit: int = 20


@dataclass
class LinkingConfig:
    """Cross-modal chunk linking via payload-based adjacency lists."""
    enabled: bool = True
    max_related_per_chunk: int = 3
    max_total_expansion: int = 10
    expansion_penalty: float = 0.9
    similarity_threshold: float = 0.70  # Conservative; calibrate via distribution analyzer
    timestamp_overlap_seconds: float = 5.0


@dataclass
class CLIPConfig:
    """CLIP visual embedding for cross-modal image search."""
    enabled: bool = True
    enabled_for_query: bool = True   # Phase 4: allow image-as-query
    model_name: str = "clip-ViT-B-32"
    collection_name: str = "image_visual_embeddings"
    vector_size: int = 512
    distance: str = "Cosine"
    device: str = "cpu"           # "cpu" or "cuda"
    max_image_results: int = 5
    modality_penalty: float = 0.95  # Slight penalty to avoid visual noise
    image_embedding_timeout_ms: int = 150  # Phase 4.6: latency guardrail
    image_search_timeout_ms: int = 50      # Phase 4.6: latency guardrail
    visual_keywords: list[str] = field(default_factory=lambda: [
        "chart", "diagram", "screenshot", "image", "graph", "figure",
        "photo", "picture", "table", "illustration", "plot", "ui",
        "interface", "visual", "slide", "drawing", "map",
    ])


@dataclass
class PageEmbeddingsConfig:
    activation_threshold: int = 50  # pages
    fallback_similarity: float = 0.25
    fallback_min_pages: int = 2


@dataclass
class LLMConfig:
    model_path: str = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    context_window: int = 4096
    max_new_tokens: int = 768
    temperature: float = 0.3
    gpu_layers: int = -1
    n_batch: int = 512
    stop_tokens: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])


@dataclass
class SLMConfig:
    """Optional small language model for query rewrite / context compression / verification."""
    enabled: bool = False
    model_path: str = "models/llm/slm.gguf"
    context_window: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.3
    gpu_layers: int = 0


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "rag_chunks"
    vector_size: int = 384
    distance: str = "Cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 64


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_upload_size_mb: int = 100


@dataclass
class PathsConfig:
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    index_dir: str = "data/index"
    logs_dir: str = "data/logs"
    models_dir: str = "models"


@dataclass
class WorkerConfig:
    document_workers: int = 2
    ocr_workers: int = 1
    audio_workers: int = 1
    task_timeout: int = 300  # seconds


@dataclass
class Settings:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    page_embeddings: PageEmbeddingsConfig = field(default_factory=PageEmbeddingsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    slm: SLMConfig = field(default_factory=SLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    log_level: str = "INFO"
    log_json: bool = True


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides. Format: RAG_SECTION__KEY=value."""
    prefix = "RAG_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        if len(parts) == 2:
            section, key = parts
            if section in config and isinstance(config[section], dict):
                # Try to preserve type
                existing = config[section].get(key)
                if isinstance(existing, int):
                    config[section][key] = int(env_value)
                elif isinstance(existing, float):
                    config[section][key] = float(env_value)
                elif isinstance(existing, bool):
                    config[section][key] = env_value.lower() in ("true", "1", "yes")
                else:
                    config[section][key] = env_value
    return config


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Load settings from config.yaml with env variable overrides.

    Args:
        config_path: Path to config.yaml. Defaults to project root.

    Returns:
        Populated Settings dataclass.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    config_dict: dict = {}

    if path.exists():
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        logger.info("Loaded config from: %s", path)
    else:
        logger.warning("Config file not found at %s. Using defaults.", path)

    # Apply environment overrides
    config_dict = _apply_env_overrides(config_dict)

    # Map to dataclasses
    settings = Settings(
        chunking=ChunkingConfig(**config_dict.get("chunking", {})),
        retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
        memory=MemoryConfig(**config_dict.get("memory", {})),
        linking=LinkingConfig(**config_dict.get("linking", {})),
        clip=CLIPConfig(**config_dict.get("clip", {})),
        page_embeddings=PageEmbeddingsConfig(**config_dict.get("page_embeddings", {})),
        llm=LLMConfig(**config_dict.get("llm", {})),
        slm=SLMConfig(**config_dict.get("slm", {})),
        qdrant=QdrantConfig(**config_dict.get("qdrant", {})),
        server=ServerConfig(**config_dict.get("server", {})),
        paths=PathsConfig(**config_dict.get("paths", {})),
        workers=WorkerConfig(**config_dict.get("workers", {})),
        log_level=config_dict.get("log_level", "INFO"),
        log_json=config_dict.get("log_json", True),
    )

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return load_settings()
