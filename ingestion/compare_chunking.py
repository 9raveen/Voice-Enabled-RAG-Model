"""
Phase 2, Step 3 — Run all four chunking strategies against the real
corpus and compare basic stats (chunk count, avg/median size).
Retrieval-quality comparison comes in the next step, after this
sanity check confirms the strategies behave as expected.
"""

import json
import statistics
from chunking_stratergies import (
    strategy_a_baseline,
    strategy_b_longtail_split,
    strategy_c_metadata_enriched,
    strategy_d_adaptive,
)

CORPUS_PATH = "data/corpus.jsonl"
EVAL_PATH = "data/eval_queries.jsonl"


def load_corpus():
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def build_passage_metadata():
    """Derive per-passage metadata (e.g. which query_types reference it)
    from the eval_queries file, since that's where query_type lives."""
    passage_meta = {}
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            for pid in q["candidate_passage_ids"]:
                if pid not in passage_meta:
                    passage_meta[pid] = {"query_types": set()}
                passage_meta[pid]["query_types"].add(q["query_type"])

    # convert sets to lists for JSON-serializability later
    for pid in passage_meta:
        passage_meta[pid]["query_types"] = list(passage_meta[pid]["query_types"])

    return passage_meta


def report(name: str, chunks: list[dict]):
    lengths = [len(c["text"]) for c in chunks]
    print(f"\n--- {name} ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {statistics.mean(lengths):.1f}")
    print(f"Median chunk length: {statistics.median(lengths)}")
    print(f"Max chunk length: {max(lengths)}")


def compare():
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} source passages")

    passage_meta = build_passage_metadata()
    print(f"Built metadata for {len(passage_meta)} passages")

    a = strategy_a_baseline(corpus)
    report("Strategy A - Baseline (no split)", a)

    b = strategy_b_longtail_split(corpus, threshold=800)
    report("Strategy B - Long-tail split (threshold=800)", b)

    c = strategy_c_metadata_enriched(corpus, passage_meta)
    report("Strategy C - Metadata-enriched", c)

    d = strategy_d_adaptive(corpus, passage_meta, threshold=800)
    report("Strategy D - Adaptive (split + metadata)", d)


if __name__ == "__main__":
    compare()