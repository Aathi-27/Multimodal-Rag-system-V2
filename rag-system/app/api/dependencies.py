"""
Dependency Injection - Shared service instances for FastAPI endpoints.

Provides singleton access to:
- VectorStore (Qdrant)
- BM25Store
- HybridRetriever
- LLMEngine
- ModelManager
- EmbeddingModel
- TaskQueue
- MetricsCollector
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config.settings import get_settings


@lru_cache(maxsize=1)
def get_vector_store():
    from app.retrieval.vector_store import VectorStore
    store = VectorStore()
    return store


@lru_cache(maxsize=1)
def get_model_registry():
    from app.models.model_registry import ModelRegistry
    settings = get_settings()
    return ModelRegistry(Path(settings.paths.models_dir))


@lru_cache(maxsize=1)
def get_model_manager():
    from app.models.model_manager import ModelManager
    registry = get_model_registry()
    return ModelManager(registry)


@lru_cache(maxsize=1)
def get_embedding_model():
    from app.models.embeddings import EmbeddingModel
    return EmbeddingModel(get_model_manager())


@lru_cache(maxsize=1)
def get_bm25_store():
    from app.retrieval.bm25_store import BM25Store
    settings = get_settings()
    index_dir = Path(settings.paths.index_dir) / "current" / "bm25"
    store = BM25Store(index_dir)
    store.load()
    return store


@lru_cache(maxsize=1)
def get_reranker():
    from app.retrieval.reranker import Reranker
    return Reranker(get_model_manager())


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    from app.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(
        vector_store=get_vector_store(),
        bm25_store=get_bm25_store(),
        reranker=get_reranker(),
        image_visual_store=get_image_visual_store(),
        clip_encoder=get_clip_encoder(),
    )


@lru_cache(maxsize=1)
def get_llm_engine():
    from app.generation.llm_engine import LLMEngine
    engine = LLMEngine()
    return engine


@lru_cache(maxsize=1)
def get_analytics_tracker():
    from app.observability.analytics import AnalyticsTracker
    settings = get_settings()
    return AnalyticsTracker(Path(settings.paths.data_dir))


@lru_cache(maxsize=1)
def get_query_store():
    from app.observability.query_store import QueryStore
    settings = get_settings()
    return QueryStore(Path(settings.paths.data_dir))


@lru_cache(maxsize=1)
def get_index_manager():
    from app.versioning.index_manager import IndexManager
    settings = get_settings()
    return IndexManager(Path(settings.paths.index_dir))


@lru_cache(maxsize=1)
def get_metrics():
    from app.observability.metrics import metrics
    return metrics


@lru_cache(maxsize=1)
def get_query_cache():
    from app.retrieval.query_cache import QueryCache
    return QueryCache(max_size=128, ttl_seconds=300.0)


@lru_cache(maxsize=1)
def get_cost_tracker():
    from app.observability.cost_tracker import CostTracker
    return CostTracker()


@lru_cache(maxsize=1)
def get_slm_engine():
    from app.generation.slm_engine import SLMEngine
    engine = SLMEngine()
    return engine


@lru_cache(maxsize=1)
def get_failure_diagnoser():
    from app.observability.failure_diagnosis import FailureDiagnoser
    return FailureDiagnoser()


@lru_cache(maxsize=1)
def get_corpus_coverage_analyzer():
    from app.observability.corpus_coverage import CorpusCoverageAnalyzer
    return CorpusCoverageAnalyzer(get_query_store(), get_vector_store())


@lru_cache(maxsize=1)
def get_embedding_quality_checker():
    from app.observability.embedding_quality import EmbeddingQualityChecker
    return EmbeddingQualityChecker(get_vector_store(), get_embedding_model())


@lru_cache(maxsize=1)
def get_experiment_engine():
    from app.experimentation.experiment_engine import ExperimentEngine
    return ExperimentEngine(get_hybrid_retriever(), get_embedding_model())


@lru_cache(maxsize=1)
def get_file_registry():
    from app.observability.file_registry import FileRegistry
    settings = get_settings()
    registry = FileRegistry(Path(settings.paths.data_dir))
    # Backfill any existing uploads not yet tracked
    registry.backfill_existing(Path(settings.paths.uploads_dir))
    return registry


@lru_cache(maxsize=1)
def get_chunk_linker():
    from app.retrieval.chunk_linker import ChunkLinker
    return ChunkLinker(get_vector_store())


@lru_cache(maxsize=1)
def get_clip_encoder():
    """Get CLIP visual encoder (lazy-loaded, returns None-safe wrapper if disabled)."""
    from app.models.clip_encoder import CLIPEncoder
    return CLIPEncoder(get_model_manager())


@lru_cache(maxsize=1)
def get_image_visual_store():
    """Get the separate Qdrant collection for CLIP visual embeddings."""
    from app.retrieval.image_visual_store import ImageVisualStore
    store = ImageVisualStore()
    store.ensure_collection()
    return store


@lru_cache(maxsize=1)
def get_image_query_retriever():
    """Get the image-as-query retriever (Phase 4: Strategy A + OCR fallback)."""
    from app.retrieval.image_query_retriever import ImageQueryRetriever
    return ImageQueryRetriever(
        image_visual_store=get_image_visual_store(),
        clip_encoder=get_clip_encoder(),
        vector_store=get_vector_store(),
        reranker=get_reranker(),
        hybrid_retriever=get_hybrid_retriever(),
        embedding_model=get_embedding_model(),
    )
