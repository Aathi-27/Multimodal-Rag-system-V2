"""
Initialize Index - Create the initial versioned index structure.

Creates:
  data/index/v1.0.0/qdrant/
  data/index/v1.0.0/bm25/
  data/index/current -> v1.0.0

Also initializes:
- Qdrant collection with correct settings
- Empty BM25 index with checksum

Usage:
    python scripts/init_index.py [--version v1.0.0] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_SYSTEM_DIR = PROJECT_ROOT / "rag-system"
sys.path.insert(0, str(RAG_SYSTEM_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def init_index(version: str = "v1.0.0", force: bool = False) -> None:
    """Initialize a versioned index structure."""
    from app.config.settings import get_settings
    from app.versioning.index_manager import IndexManager

    settings = get_settings()
    index_root = PROJECT_ROOT / settings.paths.index_dir
    manager = IndexManager(index_root)

    # Check if version exists
    if version in manager.list_versions():
        if force:
            logger.warning("Force flag set. Deleting existing version %s", version)
            # Unlink current if it points to this version
            if manager.current_version == version:
                link = manager.current_link
                if link.exists():
                    link.unlink()
            import shutil
            shutil.rmtree(index_root / version)
        else:
            logger.info("Index version %s already exists. Use --force to recreate.", version)
            return

    # Create version directory
    manager.create_version(version)
    manager.switch_to(version)
    logger.info("Index version %s created and set as current.", version)

    # Initialize Qdrant collection
    try:
        init_qdrant_collection(settings)
    except Exception as e:
        logger.warning("Could not initialize Qdrant collection: %s", e)
        logger.info("Qdrant will be initialized when the server starts.")

    logger.info("Index initialization complete.")


def init_qdrant_collection(settings) -> None:
    """Create the Qdrant collection with correct configuration."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

    client = QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
    )

    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant.collection_name in collections:
        logger.info("Qdrant collection '%s' already exists.", settings.qdrant.collection_name)
        return

    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
    }

    client.create_collection(
        collection_name=settings.qdrant.collection_name,
        vectors_config=VectorParams(
            size=settings.qdrant.vector_size,
            distance=distance_map.get(settings.qdrant.distance, Distance.COSINE),
        ),
        hnsw_config=HnswConfigDiff(
            m=settings.qdrant.hnsw_m,
            ef_construct=settings.qdrant.hnsw_ef_construct,
        ),
    )
    logger.info(
        "Created Qdrant collection '%s' (size=%d, distance=%s, m=%d, ef_construct=%d)",
        settings.qdrant.collection_name,
        settings.qdrant.vector_size,
        settings.qdrant.distance,
        settings.qdrant.hnsw_m,
        settings.qdrant.hnsw_ef_construct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize versioned index")
    parser.add_argument("--version", default="v1.0.0", help="Version string")
    parser.add_argument("--force", action="store_true", help="Force recreate if exists")
    args = parser.parse_args()

    init_index(version=args.version, force=args.force)


if __name__ == "__main__":
    main()
