"""
Phase 4.0 — Multimodal Baseline Snapshot (pre image-query).

Captures the complete system state BEFORE enabling image-as-query:
  • All Phase 0 metrics (Qdrant, BM25, config, uploads)
  • CLIP subsystem state (image_visual_embeddings collection)
  • Image branch activation rate (from query history)
  • GPU memory usage
  • P95 latency estimate
  • Recall@5 / Recall@10 / MRR (from annotated queries)
  • Index sizes

Writes to data/baseline_snapshots/baseline_vX_multimodal_pre_image_query.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag-system"))

from app.config.settings import get_settings


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


def snapshot_qdrant_text(settings) -> dict:
    """Capture text (BGE) Qdrant collection state."""
    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    coll = settings.qdrant.collection_name
    try:
        info = client.get_collection(coll)
        return {
            "collection_name": coll,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "vector_size": info.config.params.vectors.size if info.config and info.config.params else None,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e), "status": "unreachable"}


def snapshot_qdrant_clip(settings) -> dict:
    """Capture CLIP visual embeddings collection state."""
    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    coll = settings.clip.collection_name
    try:
        collections = [c.name for c in client.get_collections().collections]
        if coll not in collections:
            return {"status": "not_initialized", "collection": coll, "image_count": 0}
        info = client.get_collection(coll)
        return {
            "collection_name": coll,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size if info.config and info.config.params else None,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e), "status": "unreachable"}


def snapshot_bm25(settings) -> dict:
    bm25_dir = Path(settings.paths.index_dir) / "current" / "bm25"
    if not bm25_dir.exists():
        return {"status": "not_found"}
    return {"status": "present", "size_mb": _dir_size_mb(bm25_dir)}


def snapshot_gpu() -> dict:
    """Capture GPU memory usage."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "gpu_name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
    except Exception as e:
        return {"error": str(e)}
    return {"status": "not_available"}


def snapshot_query_history_metrics(settings) -> dict:
    """Extract recall/MRR and image branch activation from query history."""
    try:
        from app.observability.query_store import QueryStore
        store = QueryStore(Path(settings.paths.data_dir))
        queries = store.list_queries(limit=500)

        total = len(queries)
        if total == 0:
            return {"total_queries": 0, "note": "No query history available"}

        latencies = [q.get("total_latency", 0) for q in queries if q.get("total_latency", 0) > 0]
        retrieval_latencies = [q.get("retrieval_latency", 0) for q in queries if q.get("retrieval_latency", 0) > 0]

        # Recall/MRR from annotated queries
        recall_at_5_values = []
        recall_at_10_values = []
        mrr_values = []
        for q in queries:
            recall = q.get("recall_metrics", {})
            if recall:
                if recall.get("recall_at_5") is not None:
                    recall_at_5_values.append(recall["recall_at_5"])
                if recall.get("recall_at_10") is not None:
                    recall_at_10_values.append(recall["recall_at_10"])
                if recall.get("mrr") is not None:
                    mrr_values.append(recall["mrr"])

        # Image branch activation rate (from debug info)
        image_activations = 0
        queries_with_debug = 0
        for q in queries:
            debug = q.get("debug_info", {})
            if debug:
                queries_with_debug += 1
                img_info = debug.get("image_branch_info", {})
                if img_info.get("injected_count", 0) > 0:
                    image_activations += 1

        p95_latency = None
        if latencies:
            latencies.sort()
            p95_idx = int(len(latencies) * 0.95)
            p95_latency = round(latencies[min(p95_idx, len(latencies) - 1)] * 1000, 2)

        p95_retrieval = None
        if retrieval_latencies:
            retrieval_latencies.sort()
            p95_idx = int(len(retrieval_latencies) * 0.95)
            p95_retrieval = round(retrieval_latencies[min(p95_idx, len(retrieval_latencies) - 1)] * 1000, 2)

        return {
            "total_queries": total,
            "p95_total_latency_ms": p95_latency,
            "p95_retrieval_latency_ms": p95_retrieval,
            "avg_recall_at_5": round(sum(recall_at_5_values) / len(recall_at_5_values), 4) if recall_at_5_values else None,
            "avg_recall_at_10": round(sum(recall_at_10_values) / len(recall_at_10_values), 4) if recall_at_10_values else None,
            "avg_mrr": round(sum(mrr_values) / len(mrr_values), 4) if mrr_values else None,
            "annotated_queries": len(recall_at_5_values),
            "image_branch_activation_rate": round(image_activations / queries_with_debug, 4) if queries_with_debug > 0 else 0.0,
            "queries_with_debug": queries_with_debug,
        }
    except Exception as e:
        return {"error": str(e)}


def snapshot_config(settings) -> dict:
    config_path = PROJECT_ROOT / "rag-system" / "config.yaml"
    config_hash = _md5(config_path) if config_path.exists() else "missing"
    return {
        "config_hash": config_hash,
        "retrieval": {
            "vector_top_k": settings.retrieval.vector_top_k,
            "bm25_top_k": settings.retrieval.bm25_top_k,
            "rrf_k": settings.retrieval.rrf_k,
            "rerank_count": settings.retrieval.rerank_count,
            "rerank_threshold": settings.retrieval.rerank_threshold,
        },
        "clip": {
            "enabled": settings.clip.enabled,
            "max_image_results": settings.clip.max_image_results,
            "modality_penalty": settings.clip.modality_penalty,
        },
        "linking": {
            "enabled": settings.linking.enabled,
            "max_total_expansion": settings.linking.max_total_expansion,
            "expansion_penalty": settings.linking.expansion_penalty,
        },
    }


def snapshot_index_sizes(settings) -> dict:
    idx_dir = Path(settings.paths.index_dir)
    uploads_dir = Path(settings.paths.uploads_dir)
    return {
        "index_total_mb": _dir_size_mb(idx_dir),
        "uploads_total_mb": _dir_size_mb(uploads_dir),
    }


def main():
    print("=" * 60)
    print("  Phase 4.0 — Multimodal Baseline Snapshot")
    print("  (pre image-query)")
    print("=" * 60)

    settings = get_settings()

    snapshot = {
        "snapshot_name": "baseline_vX_multimodal_pre_image_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "4.0",
    }

    print("\n[1/7] Qdrant text collection...")
    snapshot["qdrant_text"] = snapshot_qdrant_text(settings)
    print(f"       → {snapshot['qdrant_text'].get('points_count', '?')} chunks")

    print("[2/7] Qdrant CLIP collection...")
    snapshot["qdrant_clip"] = snapshot_qdrant_clip(settings)
    print(f"       → {snapshot['qdrant_clip'].get('points_count', '?')} images")

    print("[3/7] BM25 index...")
    snapshot["bm25"] = snapshot_bm25(settings)
    print(f"       → {snapshot['bm25'].get('status')}")

    print("[4/7] GPU state...")
    snapshot["gpu"] = snapshot_gpu()
    print(f"       → {snapshot['gpu'].get('gpu_name', 'N/A')}")

    print("[5/7] Query metrics (recall, latency, image branch)...")
    snapshot["query_metrics"] = snapshot_query_history_metrics(settings)
    print(f"       → {snapshot['query_metrics'].get('total_queries', 0)} queries analyzed")

    print("[6/7] Configuration snapshot...")
    snapshot["config"] = snapshot_config(settings)
    print(f"       → hash: {snapshot['config']['config_hash'][:12]}...")

    print("[7/7] Index sizes...")
    snapshot["index_sizes"] = snapshot_index_sizes(settings)
    print(f"       → index: {snapshot['index_sizes']['index_total_mb']} MB")

    # Write snapshot
    out_dir = Path(settings.paths.data_dir) / "baseline_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"baseline_vX_multimodal_pre_image_query_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\n✓ Snapshot: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    print("=" * 60)
    return snapshot


if __name__ == "__main__":
    main()
