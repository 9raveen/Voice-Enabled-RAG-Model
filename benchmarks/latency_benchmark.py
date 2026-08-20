"""
Phase 8 — Latency benchmark: P50/P70/P100 across real queries,
with per-stage breakdown, run against the already-warm pipeline
(models loaded once, not per-query).
"""

import json
import random
import time
import sys
import os
import time as time_module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from pipeline import run_pipeline

EVAL_PATH = "../data/eval_queries.jsonl"
NUM_QUERIES = 20
RANDOM_SEED = 7  # different seed from Phase 7, avoid re-testing identical set


def load_queries():
    queries = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["relevant_passage_ids"]:
                queries.append(q["query"])
    random.seed(RANDOM_SEED)
    return random.sample(queries, min(NUM_QUERIES, len(queries)))


def percentile(sorted_list, p):
    idx = int(len(sorted_list) * p / 100)
    return sorted_list[min(idx, len(sorted_list) - 1)]


def main():
    queries = load_queries()
    print(f"Benchmarking {len(queries)} queries...")

    # Warm-up call to load models before timing starts
    print("Warm-up call (excluded from stats)...")
    run_pipeline(queries[0])

    totals, retrievals, generations, groundings = [], [], [], []

    for i, q in enumerate(queries):
        result = run_pipeline(q)
        time_module.sleep(13)  # 5 RPM = max 1 request per 12s; 13s adds a safety margin
        totals.append(result.latency_ms.get("total_ms", 0))
        retrievals.append(result.latency_ms.get("retrieval_ms", 0))
        generations.append(result.latency_ms.get("generation_ms", 0))
        groundings.append(result.latency_ms.get("grounding_check_ms", 0))
        print(f"  [{i+1}/{len(queries)}] {result.latency_ms.get('total_ms', 0):.0f}ms")

    totals.sort()
    retrievals.sort()
    generations.sort()
    groundings.sort()

    print(f"\n=== Total pipeline latency (n={len(totals)}) ===")
    print(f"P50: {percentile(totals, 50):.0f}ms")
    print(f"P70: {percentile(totals, 70):.0f}ms")
    print(f"P100 (max): {percentile(totals, 100):.0f}ms")
    print(f"Min: {min(totals):.0f}ms")

    print(f"\n=== Per-stage P50 ===")
    print(f"Retrieval P50: {percentile(retrievals, 50):.0f}ms")
    print(f"Generation P50: {percentile(generations, 50):.0f}ms")
    print(f"Grounding check P50: {percentile(groundings, 50):.0f}ms")

    results = {
        "n": len(totals),
        "total_ms": {"p50": percentile(totals, 50), "p70": percentile(totals, 70), "p100": percentile(totals, 100), "min": min(totals)},
        "retrieval_ms": {"p50": percentile(retrievals, 50)},
        "generation_ms": {"p50": percentile(generations, 50)},
        "grounding_ms": {"p50": percentile(groundings, 50)},
    }
    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmarks/latency_results.json")


if __name__ == "__main__":
    main()