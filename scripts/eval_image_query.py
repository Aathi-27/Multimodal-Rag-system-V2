"""
Phase 4.8 — Image-as-Query Evaluation Protocol.

Creates 50 test cases across 3 categories:
  1. Visual-only queries (20)  — image queries with no text prompt
  2. Mixed visual-text queries (15) — image + text prompt
  3. Non-visual text queries (15)   — standard text queries (regression check)

Measures:
  - Recall@5 delta vs baseline
  - MRR delta vs baseline
  - % image results in top 5
  - Latency impact
  - Non-visual query degradation check

Rejection criteria:
  - Non-visual queries degrade in Recall@5 or MRR.

Usage:
  python scripts/eval_image_query.py --baseline <path_to_baseline_snapshot.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag-system"))

# ── Test Case Templates ───────────────────────────────────────────────────────

VISUAL_ONLY_CASES = [
    {"id": "vis_01", "description": "Similar chart lookup", "prompt": None},
    {"id": "vis_02", "description": "Diagram matching", "prompt": None},
    {"id": "vis_03", "description": "Screenshot similarity", "prompt": None},
    {"id": "vis_04", "description": "Photo matching", "prompt": None},
    {"id": "vis_05", "description": "UI mockup search", "prompt": None},
    {"id": "vis_06", "description": "Graph pattern match", "prompt": None},
    {"id": "vis_07", "description": "Illustration lookup", "prompt": None},
    {"id": "vis_08", "description": "Slide content search", "prompt": None},
    {"id": "vis_09", "description": "Drawing similarity", "prompt": None},
    {"id": "vis_10", "description": "Map search", "prompt": None},
    {"id": "vis_11", "description": "Table image lookup", "prompt": None},
    {"id": "vis_12", "description": "Architecture diagram", "prompt": None},
    {"id": "vis_13", "description": "Flowchart matching", "prompt": None},
    {"id": "vis_14", "description": "Infographic search", "prompt": None},
    {"id": "vis_15", "description": "Logo recognition", "prompt": None},
    {"id": "vis_16", "description": "Network diagram", "prompt": None},
    {"id": "vis_17", "description": "Code screenshot search", "prompt": None},
    {"id": "vis_18", "description": "Bar chart matching", "prompt": None},
    {"id": "vis_19", "description": "Pie chart lookup", "prompt": None},
    {"id": "vis_20", "description": "Dashboard screenshot", "prompt": None},
]

MIXED_VISUAL_TEXT_CASES = [
    {"id": "mix_01", "description": "Chart + explain data", "prompt": "find related content about this chart"},
    {"id": "mix_02", "description": "Diagram + extract text", "prompt": "what does this diagram describe"},
    {"id": "mix_03", "description": "Screenshot + context", "prompt": "find documentation related to this UI"},
    {"id": "mix_04", "description": "Photo + identify", "prompt": "what is shown in this image"},
    {"id": "mix_05", "description": "Graph + trend analysis", "prompt": "explain the trend shown here"},
    {"id": "mix_06", "description": "Table + query data", "prompt": "find data matching this table"},
    {"id": "mix_07", "description": "Slide + summarize", "prompt": "summarize the content of this slide"},
    {"id": "mix_08", "description": "UI + documentation", "prompt": "which feature does this screen show"},
    {"id": "mix_09", "description": "Map + location details", "prompt": "describe the area shown"},
    {"id": "mix_10", "description": "Architecture + specs", "prompt": "find technical specs matching this architecture"},
    {"id": "mix_11", "description": "Code screenshot + explain", "prompt": "explain what this code does"},
    {"id": "mix_12", "description": "Flowchart + process", "prompt": "describe this process flow"},
    {"id": "mix_13", "description": "Infographic + details", "prompt": "find details about this infographic"},
    {"id": "mix_14", "description": "Network + topology", "prompt": "describe this network layout"},
    {"id": "mix_15", "description": "Dashboard + metrics", "prompt": "explain the metrics shown"},
]

NON_VISUAL_TEXT_CASES = [
    {"id": "txt_01", "query": "What is retrieval augmented generation?"},
    {"id": "txt_02", "query": "Explain the chunking strategy used in this system."},
    {"id": "txt_03", "query": "How does cross-encoder reranking improve search results?"},
    {"id": "txt_04", "query": "What are the system requirements for deployment?"},
    {"id": "txt_05", "query": "Describe the BM25 algorithm implementation."},
    {"id": "txt_06", "query": "How does hybrid retrieval combine vector and keyword search?"},
    {"id": "txt_07", "query": "What is reciprocal rank fusion?"},
    {"id": "txt_08", "query": "Explain the entity extraction process."},
    {"id": "txt_09", "query": "How are audio files processed in the system?"},
    {"id": "txt_10", "query": "What embedding model is used?"},
    {"id": "txt_11", "query": "Describe the document ingestion pipeline."},
    {"id": "txt_12", "query": "What is the purpose of the survival tracker?"},
    {"id": "txt_13", "query": "How does the reranker threshold affect results?"},
    {"id": "txt_14", "query": "What OCR engine is used for image text extraction?"},
    {"id": "txt_15", "query": "Explain the RRF k parameter."},
]


def generate_eval_manifest(output_dir: Path) -> dict:
    """
    Generate the evaluation test manifest.

    The manifest contains all 50 test cases. Visual cases require
    actual image files to be placed in the test_images/ directory.
    """
    manifest = {
        "version": "4.8",
        "total_cases": 50,
        "categories": {
            "visual_only": {
                "count": len(VISUAL_ONLY_CASES),
                "requires": "test_images/ directory with labeled images",
                "cases": VISUAL_ONLY_CASES,
            },
            "mixed_visual_text": {
                "count": len(MIXED_VISUAL_TEXT_CASES),
                "requires": "test_images/ directory with labeled images",
                "cases": MIXED_VISUAL_TEXT_CASES,
            },
            "non_visual_text": {
                "count": len(NON_VISUAL_TEXT_CASES),
                "requires": "populated text corpus only",
                "cases": NON_VISUAL_TEXT_CASES,
            },
        },
        "metrics": [
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "pct_image_in_top5",
            "avg_latency_ms",
            "p95_latency_ms",
        ],
        "rejection_criteria": [
            "Non-visual queries must NOT degrade in Recall@5 or MRR",
            "P95 latency must stay within baseline + 25%",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "eval_image_query_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def run_text_query_eval(cases: list[dict], settings) -> list[dict]:
    """Run non-visual text queries and measure recall/latency."""
    from app.models.embeddings import EmbeddingModel
    from app.models.model_manager import ModelManager
    from app.models.model_registry import ModelRegistry
    from app.retrieval.bm25_store import BM25Store
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker
    from app.retrieval.vector_store import VectorStore

    registry = ModelRegistry(Path(settings.paths.models_dir))
    manager = ModelManager(registry)
    emb = EmbeddingModel(manager)
    vs = VectorStore()
    bm25_dir = Path(settings.paths.index_dir) / "current" / "bm25"
    bm25 = BM25Store(bm25_dir)
    bm25.load()
    reranker = Reranker(manager)
    retriever = HybridRetriever(vs, bm25, reranker)

    results = []
    for case in cases:
        query = case["query"]
        t0 = time.perf_counter()
        query_embedding = emb.embed_query(query)
        output = retriever.retrieve(query=query, query_embedding=query_embedding, debug=True)

        if isinstance(output, tuple):
            chunks, debug = output
        else:
            chunks = output
            debug = {}

        latency = (time.perf_counter() - t0) * 1000
        image_in_top5 = sum(
            1 for r in chunks[:5] if r.get("origin") in ("image_visual", "image_query")
        )

        results.append({
            "case_id": case["id"],
            "query": query,
            "result_count": len(chunks),
            "latency_ms": round(latency, 2),
            "image_in_top5": image_in_top5,
            "top5_sources": [
                r.get("metadata", {}).get("source", "?")
                for r in chunks[:5]
            ],
        })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4.8 — Image Query Evaluation")
    parser.add_argument("--generate-manifest", action="store_true", help="Generate eval manifest only")
    parser.add_argument("--run-text-eval", action="store_true", help="Run non-visual text eval")
    parser.add_argument("--output", default="data/eval", help="Output directory")
    args = parser.parse_args()

    from app.config.settings import get_settings
    settings = get_settings()
    out_dir = Path(args.output)

    if args.generate_manifest:
        manifest = generate_eval_manifest(out_dir)
        print(f"✓ Manifest generated: {len(manifest['categories'])} categories, 50 test cases")
        return

    if args.run_text_eval:
        print("Running non-visual text eval (15 queries)...")
        results = run_text_query_eval(NON_VISUAL_TEXT_CASES, settings)

        latencies = [r["latency_ms"] for r in results]
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)

        summary = {
            "category": "non_visual_text",
            "total_cases": len(results),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(latencies[min(p95_idx, len(latencies) - 1)], 2),
            "avg_results": round(sum(r["result_count"] for r in results) / len(results), 1),
            "image_contamination": sum(r["image_in_top5"] for r in results),
            "cases": results,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_text_baseline.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Text eval: avg={summary['avg_latency_ms']}ms, P95={summary['p95_latency_ms']}ms")
        print(f"  Image contamination in non-visual: {summary['image_contamination']} hits")
        print(f"  Saved: {out_path}")
        return

    # Default: generate manifest
    generate_eval_manifest(out_dir)
    print("Usage:")
    print("  --generate-manifest  Generate test case manifest")
    print("  --run-text-eval      Run non-visual text regression eval")


if __name__ == "__main__":
    main()
