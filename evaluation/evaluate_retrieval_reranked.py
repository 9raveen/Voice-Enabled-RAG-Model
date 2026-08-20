"""
Phase 4, Step 2 — Retrieval + reranking evaluation.
Retrieves top-20 candidates via Qdrant (as before), then reranks them
with a cross-encoder before computing Recall@K / MRR — same metric,
same eval set, so results are directly comparable to Phase 4 Step 1.
"""

import json
import random
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

EVAL_PATH = "data/eval_queries.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "voice_rag_corpus"
NUM_EVAL_QUERIES = 300
RETRIEVE_TOP_N = 20   # candidates fetched before reranking
K_VALUES = [1, 5, 10]
RANDOM_SEED = 42


def load_eval_queries():
    queries = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["relevant_passage_ids"]:
                queries.append(q)
    return queries


def main():
    print("Loading eval queries with ground truth...")
    all_queries = load_eval_queries()
    random.seed(RANDOM_SEED)
    eval_set = random.sample(all_queries, min(NUM_EVAL_QUERIES, len(all_queries)))
    print(f"Evaluating on {len(eval_set)} sampled queries (same seed as Step 1 for fair comparison)")

    print(f"Loading embedding model {MODEL_NAME} on GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()

    print(f"Loading reranker {RERANKER_NAME} on GPU...")
    reranker = CrossEncoder(RERANKER_NAME, device="cuda", model_kwargs={"torch_dtype": "float16"})

    client = QdrantClient(url=QDRANT_URL)

    max_k = max(K_VALUES)
    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    rerank_times = []

    start = time.time()
    for q in eval_set:
        query_text = "query: " + q["query"]
        query_vec = model.encode([query_text], normalize_embeddings=True)[0]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec.tolist(),
            limit=RETRIEVE_TOP_N,
        )
        candidates = results.points
        rerank_start = time.time()

        pairs = [[q["query"], c.payload["text"]] for c in candidates]
        scores = reranker.predict(pairs, batch_size=20, show_progress_bar=False)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        rerank_times.append(time.time() - rerank_start)
        retrieved_source_ids = [c.payload["source_passage_id"] for c, _ in reranked]

        relevant_set = set(q["relevant_passage_ids"])

        rr = 0.0
        for rank, sid in enumerate(retrieved_source_ids, start=1):
            if sid in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        for k in K_VALUES:
            top_k_ids = set(retrieved_source_ids[:k])
            if top_k_ids & relevant_set:
                hits_at_k[k] += 1

    elapsed = time.time() - start
    n = len(eval_set)

    print(f"\n=== Retrieval + Reranking Evaluation ({n} queries, {elapsed:.1f}s total, "
          f"{elapsed/n*1000:.1f}ms/query avg) ===")
    for k in K_VALUES:
        print(f"Recall@{k}: {hits_at_k[k]/n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks)/n:.3f}")
    print(f"Avg reranking-only time: {sum(rerank_times)/n*1000:.1f}ms/query")


if __name__ == "__main__":
    main()