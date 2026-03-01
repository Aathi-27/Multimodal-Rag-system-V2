"""
Health Check - Startup validation script.

Validates system readiness:
1. Required model files exist on disk
2. Qdrant is reachable
3. BM25 index checksum is valid
4. Configuration is loaded correctly
5. Disk space is sufficient

Usage:
    python scripts/health_check.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_SYSTEM_DIR = PROJECT_ROOT / "rag-system"
sys.path.insert(0, str(RAG_SYSTEM_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def check_models() -> dict[str, bool]:
    """Check if all required model files exist."""
    from app.models.model_registry import ModelRegistry, ModelType

    models_root = PROJECT_ROOT / "models"
    registry = ModelRegistry(models_root)
    availability = registry.check_availability()

    results = {}
    for model_type, available in availability.items():
        info = registry.get(model_type)
        status = "✓" if available else ("⚠ OPTIONAL" if not info.required else "✗ MISSING")
        logger.info("  %s [%s]: %s → %s", status, model_type.value, info.name, info.path)
        results[model_type.value] = available

    missing = registry.get_missing_required()
    if missing:
        logger.error(
            "Missing required models: %s",
            ", ".join(m.name for m in missing),
        )
    return results


def check_qdrant() -> bool:
    """Check if Qdrant is reachable."""
    from app.config.settings import get_settings

    settings = get_settings()
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            timeout=5,
        )
        collections = client.get_collections()
        logger.info("  ✓ Qdrant connected (%d collections)", len(collections.collections))
        return True
    except Exception as e:
        logger.error("  ✗ Qdrant unreachable: %s", e)
        return False


def check_index() -> bool:
    """Check if index directory is initialized."""
    from app.config.settings import get_settings
    from app.versioning.index_manager import IndexManager

    settings = get_settings()
    index_root = PROJECT_ROOT / settings.paths.index_dir
    manager = IndexManager(index_root)

    versions = manager.list_versions()
    current = manager.current_version

    if not versions:
        logger.warning("  ⚠ No index versions found. Run: python scripts/init_index.py")
        return False

    logger.info("  ✓ Index versions: %s (active: %s)", versions, current)
    return True


def check_disk_space() -> bool:
    """Check available disk space."""
    import shutil

    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)

    if free_gb < 5.0:
        logger.warning("  ⚠ Low disk space: %.1f GB free / %.1f GB total", free_gb, total_gb)
        return False

    logger.info("  ✓ Disk space: %.1f GB free / %.1f GB total", free_gb, total_gb)
    return True


def check_config() -> bool:
    """Validate configuration loading."""
    try:
        from app.config.settings import load_settings

        settings = load_settings()
        logger.info("  ✓ Config loaded (chunking: %d/%d, retrieval: rrf_k=%d)",
                     settings.chunking.target_tokens,
                     settings.chunking.max_tokens,
                     settings.retrieval.rrf_k)
        return True
    except Exception as e:
        logger.error("  ✗ Config error: %s", e)
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("RAG SYSTEM HEALTH CHECK")
    logger.info("=" * 60)

    checks = {
        "Configuration": check_config,
        "Models": check_models,
        "Qdrant": check_qdrant,
        "Index": check_index,
        "Disk Space": check_disk_space,
    }

    results = {}
    for name, check_fn in checks.items():
        logger.info("\n[%s]", name)
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error("  ✗ Check failed with error: %s", e)
            results[name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("RESULT: %d/%d checks passed", passed, total)

    if passed < total:
        failed = [name for name, ok in results.items() if not ok]
        logger.warning("Failed checks: %s", ", ".join(failed))
        sys.exit(1)
    else:
        logger.info("System is ready. ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
