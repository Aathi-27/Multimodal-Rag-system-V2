"""
Phase 0 — Freeze Baseline Snapshot.

Captures a point-in-time snapshot of the entire system state:
  • Qdrant collection config + stats
  • BM25 index state
  • Schema (payload fields, vector config)
  • Baseline metrics: Recall@k, MRR, latency P95, index size, ingestion throughput
  • Config.yaml hash for drift detection

Writes a timestamped JSON report to data/baseline_snapshots/.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag-system"))

from app.config.settings import get_settings


def _md5(path: Path) -> str:
    """Return hex MD5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_size_mb(path: Path) -> float:
    """Recursive directory size in MB."""
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


def snapshot_qdrant(settings) -> dict:
    """Capture Qdrant collection configuration and statistics."""
    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    coll_name = settings.qdrant.collection_name

    try:
        info = client.get_collection(coll_name)
    except Exception as e:
        return {"error": str(e), "status": "unreachable"}

    # Sample payload field keys from first 5 points
    sample_fields = set()
    try:
        pts, _ = client.scroll(coll_name, limit=5, with_payload=True, with_vectors=False)
        for p in pts:
            if p.payload:
                sample_fields.update(p.payload.keys())
    except Exception:
        pass

    return {
        "collection_name": coll_name,
        "points_count": info.points_count,
        "segments_count": info.segments_count,
        "vector_size": info.config.params.vectors.size if info.config and info.config.params else None,
        "distance": str(info.config.params.vectors.distance) if info.config and info.config.params else None,
        "hnsw_m": info.config.hnsw_config.m if info.config and info.config.hnsw_config else None,
        "hnsw_ef_construct": info.config.hnsw_config.ef_construct if info.config and info.config.hnsw_config else None,
        "payload_fields_sample": sorted(sample_fields),
        "status": str(info.status),
    }


def snapshot_bm25(settings) -> dict:
    """Capture BM25 index state."""
    bm25_dir = Path(settings.paths.index_dir) / "current" / "bm25"
    if not bm25_dir.exists():
        return {"status": "not_found", "path": str(bm25_dir)}

    files = {}
    for f in bm25_dir.iterdir():
        if f.is_file():
            files[f.name] = {
                "size_bytes": f.stat().st_size,
                "md5": _md5(f),
            }

    return {
        "path": str(bm25_dir),
        "files": files,
        "total_size_mb": _dir_size_mb(bm25_dir),
        "status": "present",
    }


def snapshot_index_versions(settings) -> dict:
    """Capture index version layout."""
    idx_dir = Path(settings.paths.index_dir)
    versions = []
    for d in sorted(idx_dir.iterdir()):
        if d.is_dir() and d.name != "current":
            versions.append({
                "version": d.name,
                "size_mb": _dir_size_mb(d),
            })
    current_link = idx_dir / "current"
    current_target = None
    if current_link.is_symlink():
        current_target = str(current_link.resolve())
    elif current_link.is_dir():
        current_target = str(current_link)

    return {
        "versions": versions,
        "current": current_target,
        "index_dir_size_mb": _dir_size_mb(idx_dir),
    }


def snapshot_config(settings) -> dict:
    """Capture config.yaml hash and key parameters."""
    config_path = PROJECT_ROOT / "rag-system" / "config.yaml"
    config_hash = _md5(config_path) if config_path.exists() else "missing"

    return {
        "config_hash": config_hash,
        "chunking": {
            "target_tokens": settings.chunking.target_tokens,
            "max_tokens": settings.chunking.max_tokens,
            "overlap_tokens": settings.chunking.overlap_tokens,
        },
        "retrieval": {
            "vector_top_k": settings.retrieval.vector_top_k,
            "bm25_top_k": settings.retrieval.bm25_top_k,
            "rrf_k": settings.retrieval.rrf_k,
            "rerank_count": settings.retrieval.rerank_count,
            "rerank_threshold": settings.retrieval.rerank_threshold,
            "rerank_min_results": settings.retrieval.rerank_min_results,
        },
        "qdrant": {
            "vector_size": settings.qdrant.vector_size,
            "distance": settings.qdrant.distance,
            "hnsw_m": settings.qdrant.hnsw_m,
            "hnsw_ef_construct": settings.qdrant.hnsw_ef_construct,
            "hnsw_ef_search": settings.qdrant.hnsw_ef_search,
        },
        "llm": {
            "model_path": settings.llm.model_path,
            "context_window": settings.llm.context_window,
            "temperature": settings.llm.temperature,
            "gpu_layers": settings.llm.gpu_layers,
        },
    }


def snapshot_uploads(settings) -> dict:
    """Inventory of uploaded files."""
    uploads_dir = Path(settings.paths.uploads_dir)
    if not uploads_dir.exists():
        return {"status": "empty", "file_count": 0, "total_size_mb": 0}

    files = []
    for f in sorted(uploads_dir.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            })

    return {
        "file_count": len(files),
        "total_size_mb": _dir_size_mb(uploads_dir),
        "files": files,
    }


def estimate_metrics(settings) -> dict:
    """
    Record observable baseline metrics.

    Since we may not have ground-truth annotations yet, we record what we can:
    - Corpus size
    - Index sizes
    - Embedding latency (single warmup query)
    """
    from app.models.embeddings import EmbeddingModel
    from app.models.model_manager import ModelManager
    from app.models.model_registry import ModelRegistry

    metrics: dict = {}

    # Corpus size from Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
        info = client.get_collection(settings.qdrant.collection_name)
        metrics["corpus_chunks"] = info.points_count
    except Exception as e:
        metrics["corpus_chunks"] = f"error: {e}"

    # Embedding latency benchmark (5 queries)
    try:
        registry = ModelRegistry(Path(settings.paths.models_dir))
        manager = ModelManager(registry)
        emb = EmbeddingModel(manager)
        emb.embed_single("warmup")

        test_queries = [
            "What is retrieval augmented generation?",
            "Explain the chunking strategy.",
            "How does the reranker work?",
            "Summarize the document.",
            "What are the system requirements?",
        ]
        latencies = []
        for q in test_queries:
            t0 = time.perf_counter()
            emb.embed_query(q)
            latencies.append(time.perf_counter() - t0)

        latencies.sort()
        metrics["embedding_latency"] = {
            "avg_ms": round(sum(latencies) / len(latencies) * 1000, 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 2),
            "min_ms": round(min(latencies) * 1000, 2),
            "max_ms": round(max(latencies) * 1000, 2),
        }
    except Exception as e:
        metrics["embedding_latency"] = f"error: {e}"

    # Index sizes on disk
    index_dir = Path(settings.paths.index_dir)
    metrics["index_size_mb"] = _dir_size_mb(index_dir)
    metrics["uploads_size_mb"] = _dir_size_mb(Path(settings.paths.uploads_dir))

    return metrics


def main():
    print("=" * 60)
    print("  Phase 0 — Baseline Snapshot")
    print("=" * 60)

    settings = get_settings()

    snapshot = {
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_version": "1.0.0",
    }

    print("\n[1/6] Snapshotting Qdrant collection...")
    snapshot["qdrant"] = snapshot_qdrant(settings)
    print(f"       → {snapshot['qdrant'].get('points_count', '?')} chunks, status={snapshot['qdrant'].get('status')}")

    print("[2/6] Snapshotting BM25 index...")
    snapshot["bm25"] = snapshot_bm25(settings)
    print(f"       → {snapshot['bm25'].get('total_size_mb', 0)} MB")

    print("[3/6] Snapshotting index versions...")
    snapshot["index_versions"] = snapshot_index_versions(settings)
    print(f"       → {len(snapshot['index_versions'].get('versions', []))} versions")

    print("[4/6] Snapshotting configuration...")
    snapshot["config"] = snapshot_config(settings)
    print(f"       → config hash: {snapshot['config']['config_hash'][:12]}...")

    print("[5/6] Inventorying uploads...")
    snapshot["uploads"] = snapshot_uploads(settings)
    print(f"       → {snapshot['uploads']['file_count']} files, {snapshot['uploads']['total_size_mb']} MB")

    print("[6/6] Recording baseline metrics...")
    snapshot["metrics"] = estimate_metrics(settings)
    print(f"       → corpus: {snapshot['metrics'].get('corpus_chunks')} chunks")

    # Write snapshot
    out_dir = Path(settings.paths.data_dir) / "baseline_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_{snapshot['snapshot_id']}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\n✓ Snapshot written to: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    return snapshot


if __name__ == "__main__":
    main()
