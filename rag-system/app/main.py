"""
FastAPI Application Entry Point - Offline RAG System.

Startup sequence:
1. Load configuration from config.yaml
2. Setup structured logging
3. Initialize index versioning
4. Connect to Qdrant
5. Load BM25 index (validate checksum)
6. Register API routes
7. Start serving

Shutdown:
1. Shutdown worker processes
2. Unload models
3. Close connections
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config.settings import get_settings
from app.observability.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────
    setup_logging(
        log_dir=Path(settings.paths.logs_dir),
        level=settings.log_level,
        json_format=settings.log_json,
    )
    logger.info("=" * 60)
    logger.info("RAG System starting...")
    logger.info("=" * 60)

    # Ensure data directories exist
    for dir_path in [
        settings.paths.uploads_dir,
        settings.paths.index_dir,
        settings.paths.logs_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Initialize index versioning
    from app.versioning.index_manager import IndexManager
    index_manager = IndexManager(Path(settings.paths.index_dir))
    index_manager.ensure_initialized()
    logger.info("Index version: %s", index_manager.current_version)

    # Connect to Qdrant
    try:
        from app.api.dependencies import get_vector_store
        vs = get_vector_store()
        vs.ensure_collection()
        logger.info("Qdrant connected. Corpus size: %d", vs.count())
    except Exception as e:
        logger.warning("Qdrant connection failed: %s (will retry on first query)", e)

    # Load BM25 index
    try:
        from app.api.dependencies import get_bm25_store
        bm25 = get_bm25_store()
        if bm25.is_loaded:
            logger.info("BM25 index loaded: %d chunks", bm25.chunk_count)
        else:
            logger.info("BM25 index not found. Will be built on first ingestion.")
    except Exception as e:
        logger.warning("BM25 load skipped: %s", e)

    # Pre-load embedding model (avoid cold-start timeout on first query)
    try:
        from app.api.dependencies import get_embedding_model
        logger.info("Pre-loading embedding model...")
        emb = get_embedding_model()
        emb.embed_single("warmup")  # Force actual model load
        logger.info("Embedding model pre-loaded successfully.")
    except Exception as e:
        logger.warning("Embedding model pre-load failed: %s (will retry on first query)", e)

    logger.info("RAG System ready. Serving on %s:%d", settings.server.host, settings.server.port)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("RAG System shutting down...")

    try:
        from app.api.dependencies import get_model_manager
        get_model_manager().unload_all()
    except Exception:
        pass

    logger.info("Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="Offline RAG System",
    description="Multimodal Retrieval-Augmented Generation Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React frontend to call the API
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from app.api.routes import router as api_router
app.include_router(api_router)
