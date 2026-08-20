"""
Phase 4, Step 1 — Retrieval quality evaluation using real ground-truth
relevance labels from MSMARCO-XI (is_selected field, captured earlier
in eval_queries.jsonl).

Measures Recall@K and MRR@K for K in {1, 5, 10}.
"""

import json
import random
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

EVAL_PATH = "data/eval_queries.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "voice_rag_corpus"
NUM_EVAL_QUERIES = 300   # subsample for speed; full 4000 optional later
K_VALUES = [1, 5, 10]
RANDOM_SEED = 42


def load_eval_queries():
    queries = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["relevant_passage_ids"]:  # skip queries with no ground truth
                queries.append(q)
    return queries


def main():
    print("Loading eval queries with ground truth...")
    all_queries = load_eval_queries()
    print(f"Total queries with relevance labels: {len(all_queries)}")

    random.seed(RANDOM_SEED)
    eval_set = random.sample(all_queries, min(NUM_EVAL_QUERIES, len(all_queries)))
    print(f"Evaluating on {len(eval_set)} sampled queries")

    print(f"Loading model {MODEL_NAME} on GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()

    client = QdrantClient(url=QDRANT_URL)

    max_k = max(K_VALUES)
    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    start = time.time()
    for q in eval_set:
        query_text = "query: " + q["query"]
        query_vec = model.encode([query_text], normalize_embeddings=True)[0]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec.tolist(),
            limit=max_k,
        )

        relevant_set = set(q["relevant_passage_ids"])
        retrieved_source_ids = [p.payload["source_passage_id"] for p in results.points]

        # MRR: find rank of first relevant hit
        rr = 0.0
        for rank, sid in enumerate(retrieved_source_ids, start=1):
            if sid in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall@K for each K
        for k in K_VALUES:
            top_k_ids = set(retrieved_source_ids[:k])
            if top_k_ids & relevant_set:
                hits_at_k[k] += 1

    elapsed = time.time() - start
    n = len(eval_set)

    print(f"\n=== Retrieval Evaluation ({n} queries, {elapsed:.1f}s total, "
          f"{elapsed/n*1000:.1f}ms/query avg) ===")
    for k in K_VALUES:
        recall = hits_at_k[k] / n
        print(f"Recall@{k}: {recall:.3f}")
    print(f"MRR: {sum(reciprocal_ranks)/n:.3f}")


if __name__ == "__main__":
    main()