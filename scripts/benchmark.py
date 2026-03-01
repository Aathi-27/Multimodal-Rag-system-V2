"""
Benchmark Runner — Controlled Before→After Measurement.

Runs a set of test queries and produces a report with:
  - P50 / P95 latency
  - Average cost per query + cost per 1K
  - Confidence distribution (High / Medium / Low)
  - Hallucination risk distribution
  - Cache hit ratio (on repeat queries)
  - Business impact (time saved, ROI)

Usage:
    python scripts/benchmark.py                     # Run all queries
    python scripts/benchmark.py --queries 5         # Run N queries
    python scripts/benchmark.py --repeat            # Run twice to test cache
    python scripts/benchmark.py --output report.json # Save JSON report
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

API_URL = "http://localhost:8000"

# ── Test queries covering different complexity levels ─────────────────────────
TEST_QUERIES = [
    "What is the document about?",
    "What is the use of Final Cut Pro?",
    "Explain the key features of the software",
    "What are the main topics discussed?",
    "How does the editing process work?",
    "What tools are available for video editing?",
    "Describe the audio processing capabilities",
    "What file formats are supported?",
]


def run_single_query(query: str, max_tokens: int = 512) -> dict:
    """Run a single SSE query and collect all metrics."""
    t0 = time.perf_counter()

    resp = requests.post(
        f"{API_URL}/query",
        json={"query": query, "max_tokens": max_tokens},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )

    meta = None
    token_count = 0
    citations = []
    answer_tokens = []
    first_token_time = None

    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode()
        if not text.startswith("data: "):
            continue
        try:
            ev = json.loads(text[6:])
        except json.JSONDecodeError:
            continue

        ev_type = ev.get("type")
        if ev_type == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            token_count += 1
            answer_tokens.append(ev.get("content", ""))
        elif ev_type == "citation":
            citations.append(ev)
        elif ev_type == "meta":
            meta = ev
        elif ev_type == "done":
            break

    total_time = time.perf_counter() - t0

    return {
        "query": query,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(first_token_time, 3) if first_token_time else None,
        "tokens_generated": token_count,
        "tok_per_s": round(token_count / total_time, 1) if total_time > 0 else 0,
        "citations_count": len(citations),
        "answer_preview": "".join(answer_tokens)[:200],
        "confidence": meta.get("confidence") if meta else None,
        "cost": meta.get("cost") if meta else None,
        "hallucination": meta.get("hallucination") if meta else None,
        "cached": meta.get("cached", False) if meta else False,
    }


def run_benchmark(
    queries: list[str],
    repeat: bool = False,
) -> dict:
    """Run full benchmark and return aggregated report."""
    results = []

    print(f"\n{'='*70}")
    print(f"  RAG System Benchmark — {len(queries)} queries")
    print(f"{'='*70}\n")

    # First pass
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q[:60]}...", end="", flush=True)
        r = run_single_query(q)
        results.append(r)
        status = "CACHED" if r["cached"] else f"{r['total_time_s']}s"
        conf = r["confidence"]["level"] if r.get("confidence") else "?"
        print(f"  → {status} | {conf} confidence | {r['tokens_generated']} tok")

    # Second pass (cache test)
    if repeat:
        print(f"\n  --- Repeat pass (cache test) ---\n")
        for i, q in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {q[:60]}...", end="", flush=True)
            r = run_single_query(q)
            results.append(r)
            status = "CACHED" if r["cached"] else f"{r['total_time_s']}s"
            print(f"  → {status}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    fresh = [r for r in results if not r["cached"]]
    cached = [r for r in results if r["cached"]]

    latencies = [r["total_time_s"] for r in fresh]
    if not latencies:
        latencies = [0]

    p50 = round(statistics.median(latencies), 2)
    p95 = round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2)
    avg_lat = round(statistics.mean(latencies), 2)

    # Confidence distribution
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    for r in fresh:
        if r.get("confidence"):
            level = r["confidence"]["level"]
            conf_dist[level] = conf_dist.get(level, 0) + 1

    medium_plus = conf_dist["high"] + conf_dist["medium"]
    medium_plus_pct = round(medium_plus / len(fresh) * 100, 1) if fresh else 0

    # Hallucination distribution
    hall_dist = {"low": 0, "medium": 0, "high": 0}
    grounding_ratios = []
    for r in fresh:
        if r.get("hallucination"):
            level = r["hallucination"]["risk_level"]
            hall_dist[level] = hall_dist.get(level, 0) + 1
            grounding_ratios.append(r["hallucination"]["grounded_ratio"])

    avg_grounding = round(statistics.mean(grounding_ratios) * 100, 1) if grounding_ratios else 0

    # Cost
    costs = [r["cost"]["estimated_cost_usd"] for r in fresh if r.get("cost")]
    avg_cost = statistics.mean(costs) if costs else 0
    cost_per_1k = round(avg_cost * 1000, 4)

    # Tokens
    token_counts = [r["tokens_generated"] for r in fresh]
    avg_tokens = round(statistics.mean(token_counts), 1) if token_counts else 0

    # Cache metrics
    cache_times = [r["total_time_s"] for r in cached]
    avg_cache_time = round(statistics.mean(cache_times), 2) if cache_times else 0

    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(results),
        "fresh_queries": len(fresh),
        "cached_queries": len(cached),
        "latency": {
            "avg_s": avg_lat,
            "p50_s": p50,
            "p95_s": p95,
            "cached_avg_s": avg_cache_time,
        },
        "confidence": {
            "distribution": conf_dist,
            "medium_plus_pct": medium_plus_pct,
        },
        "hallucination": {
            "risk_distribution": hall_dist,
            "avg_grounding_pct": avg_grounding,
        },
        "cost": {
            "avg_per_query_usd": round(avg_cost, 6),
            "per_1k_queries_usd": cost_per_1k,
        },
        "generation": {
            "avg_tokens": avg_tokens,
            "avg_tok_per_s": round(statistics.mean([r["tok_per_s"] for r in fresh]) if fresh else 0, 1),
        },
        "results": results,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Queries: {len(fresh)} fresh + {len(cached)} cached")
    print(f"")
    print(f"  ┌─────────────────────────┬───────────────┐")
    print(f"  │ Metric                  │ Value         │")
    print(f"  ├─────────────────────────┼───────────────┤")
    print(f"  │ Avg Latency             │ {avg_lat:>10}s   │")
    print(f"  │ P50 Latency             │ {p50:>10}s   │")
    print(f"  │ P95 Latency             │ {p95:>10}s   │")
    print(f"  │ Cached Avg Latency      │ {avg_cache_time:>10}s   │")
    print(f"  │ Avg Tokens/Query        │ {avg_tokens:>10}     │")
    print(f"  │ Cost / 1K Queries       │ ${cost_per_1k:>9}   │")
    print(f"  │ Medium+ Confidence      │ {medium_plus_pct:>9}%   │")
    print(f"  │ Avg Grounding           │ {avg_grounding:>9}%   │")
    print(f"  │ High Hallucination Risk │ {hall_dist.get('high', 0):>10}     │")
    print(f"  └─────────────────────────┴───────────────┘")
    print(f"")

    return report


def main():
    parser = argparse.ArgumentParser(description="RAG System Benchmark Runner")
    parser.add_argument("--queries", type=int, default=None, help="Number of queries to run")
    parser.add_argument("--repeat", action="store_true", help="Run queries twice to test cache")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    queries = TEST_QUERIES[:args.queries] if args.queries else TEST_QUERIES

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: Server not healthy (status {r.status_code})")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {API_URL}. Is the server running?")
        sys.exit(1)

    report = run_benchmark(queries, repeat=args.repeat)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"  Report saved to {out_path}")

    # Also fetch business impact
    try:
        impact_resp = requests.get(f"{API_URL}/business-impact", timeout=5)
        if impact_resp.status_code == 200:
            impact = impact_resp.json()
            imp = impact.get("impact", {})
            per_q = imp.get("per_query", {})
            per_1k = imp.get("per_1k_queries", {})

            print(f"\n  ┌─────────────────────────┬───────────────┐")
            print(f"  │ Business Impact          │ Value         │")
            print(f"  ├─────────────────────────┼───────────────┤")
            print(f"  │ Manual search time       │ {per_q.get('manual_search_time_s', 0):>8}s     │")
            print(f"  │ System response time     │ {per_q.get('system_response_time_s', 0):>8}s     │")
            print(f"  │ Time saved / query       │ {per_q.get('time_saved_s', 0):>8}s     │")
            print(f"  │ Time saved %             │ {per_q.get('time_saved_pct', 0):>8}%     │")
            print(f"  │ Labor saved / 1K queries │  ${per_1k.get('labor_saved_usd', 0):>10}   │")
            print(f"  │ Net ROI / 1K queries     │  ${per_1k.get('net_roi_usd', 0):>10}   │")
            print(f"  │ ROI Multiplier           │ {per_1k.get('roi_multiplier', 'N/A'):>12}   │")
            print(f"  │ Speedup Factor           │ {imp.get('speedup_factor', 'N/A'):>12}   │")
            print(f"  └─────────────────────────┴───────────────┘")
    except Exception:
        pass

    print()


if __name__ == "__main__":
    main()
