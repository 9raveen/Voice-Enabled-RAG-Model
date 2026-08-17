"""
Phase 2, Step 1 — Analyze length distribution across the FULL corpus
(not a sample), to decide chunking thresholds with evidence.
"""

import json
import statistics

CORPUS_PATH = "data/corpus.jsonl"


def analyze():
    lengths = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lengths.append(len(row["text"]))

    lengths.sort()
    n = len(lengths)

    def percentile(p):
        idx = int(n * p / 100)
        return lengths[min(idx, n - 1)]

    print(f"Total passages: {n}")
    print(f"Min length: {lengths[0]}")
    print(f"Max length: {lengths[-1]}")
    print(f"Mean length: {statistics.mean(lengths):.1f}")
    print(f"Median (P50): {percentile(50)}")
    print(f"P75: {percentile(75)}")
    print(f"P90: {percentile(90)}")
    print(f"P95: {percentile(95)}")
    print(f"P99: {percentile(99)}")

    # How many passages exceed common chunk-size thresholds?
    for threshold in [500, 800, 1000, 1500, 2000]:
        count = sum(1 for l in lengths if l > threshold)
        pct = 100 * count / n
        print(f"Passages > {threshold} chars: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    analyze()