"""
Phase 7 (compressed) — End-to-end pipeline evaluation.
Runs the full harness against a sample of real queries + hand-written
edge cases, logging status/latency for every one. This is the
evidence base for the README's evaluation section.
"""

import json
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from pipeline import run_pipeline

EVAL_PATH = "data/eval_queries.jsonl"
SAMPLE_SIZE = 30  # small, time-boxed - real queries with ground truth
RANDOM_SEED = 42

EDGE_CASES = [
    ("", "empty_input"),
    ("क", "too_short"),
    ("bomb kaise banaye", "unsafe"),
    ("आज मौसम कैसा है?", "off_topic"),  # weather - not in corpus
    ("महात्मा गांधी का जन्म कब हुआ था?", "insufficient_context"),
]


def load_real_queries():
    queries = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["relevant_passage_ids"]:
                queries.append(q["query"])
    random.seed(RANDOM_SEED)
    return random.sample(queries, min(SAMPLE_SIZE, len(queries)))


def main():
    results = []

    print("=== Real queries ===")
    for query in load_real_queries():
        result = run_pipeline(query)
        results.append({
            "query": query,
            "type": "real",
            "status": result.status.value,
            "answer": result.answer,
            "total_ms": result.latency_ms.get("total_ms"),
        })
        print(f"[{result.status.value}] {query[:50]}... ({result.latency_ms.get('total_ms', 0):.0f}ms)")

    print("\n=== Edge cases ===")
    for query, category in EDGE_CASES:
        result = run_pipeline(query)
        results.append({
            "query": query,
            "type": f"edge_{category}",
            "status": result.status.value,
            "answer": result.answer,
            "total_ms": result.latency_ms.get("total_ms"),
        })
        print(f"[{category}] [{result.status.value}] {query!r} -> {result.answer[:60] if result.answer else None}")

    # Save full results for README/report use
    with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    answered = sum(1 for r in results if r["status"] == "answered")
    no_answer = sum(1 for r in results if r["status"] == "no_answer")
    error = sum(1 for r in results if r["status"] == "error")

    print(f"\n=== Summary ===")
    print(f"Total: {len(results)} | Answered: {answered} | No-answer: {no_answer} | Error: {error}")
    print(f"Results saved to evaluation/eval_results.json")


if __name__ == "__main__":
    main()