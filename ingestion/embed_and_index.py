"""
Phase 3, Step 2 — Full offline indexing pipeline.
corpus.jsonl -> Strategy D chunking -> embed (multilingual-e5-large) -> Qdrant

This is the OFFLINE pipeline. Run once; the online query pipeline
(Phase 4+) reads from the persisted Qdrant collection this creates.
"""

import json
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from chunking_stratergies import strategy_d_adaptive

CORPUS_PATH = "data/corpus.jsonl"
EVAL_PATH = "data/eval_queries.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-large"
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "voice_rag_corpus"
BATCH_SIZE = 128


def load_corpus():
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def build_passage_metadata():
    passage_meta = {}
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            for pid in q["candidate_passage_ids"]:
                if pid not in passage_meta:
                    passage_meta[pid] = {"query_types": set()}
                passage_meta[pid]["query_types"].add(q["query_type"])
    for pid in passage_meta:
        passage_meta[pid]["query_types"] = list(passage_meta[pid]["query_types"])
    return passage_meta


def main():
    print("Loading corpus and metadata...")
    corpus = load_corpus()
    passage_meta = build_passage_metadata()

    print("Running Strategy D (adaptive) chunking...")
    chunks = strategy_d_adaptive(corpus, passage_meta, threshold=800)
    print(f"Total chunks to index: {len(chunks)}")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.half()  # FP16 — roughly 2x faster on GPU, negligible quality loss for retrieval
    print(f"Model device: {model.device}")

    print(f"Initializing persistent Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    print(f"Embedding and indexing {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    start = time.time()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = ["passage: " + c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = []
        for j, chunk in enumerate(batch):
            points.append(PointStruct(
                id=i + j,
                vector=embeddings[j].tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "source_passage_id": chunk["source_passage_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                },
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)

        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - start
            done = i + len(batch)
            print(f"  {done}/{len(chunks)} chunks indexed ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start
    print(f"\nDone. Indexed {len(chunks)} chunks in {total_time:.1f}s "
          f"({len(chunks)/total_time:.1f} chunks/sec)")

    count = client.count(collection_name=COLLECTION_NAME)
    print(f"Verified collection count: {count.count}")


if __name__ == "__main__":
    main()