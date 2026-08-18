"""
Phase 3, Step 1 — Sanity check: embedding model + local/embedded Qdrant.
No server, no Docker — Qdrant runs in-process and persists to ./qdrant_data.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

MODEL_NAME = "intfloat/multilingual-e5-large"
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "voice_rag_test"


def main():
    print(f"Loading embedding model: {MODEL_NAME} (first run downloads it, ~1-2GB)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # e5 models require a "passage: " / "query: " prefix convention — this
    # matters for retrieval quality, not optional formatting.
    test_passages = [
        "passage: कॉर्पोरेशन एक कंपनी होती है जो कानूनी रूप से अपने मालिकों से अलग होती है।",
        "passage: भारत की राजधानी नई दिल्ली है।",
        "passage: महात्मा गांधी का जन्म 2 अक्टूबर 1869 को हुआ था।",
    ]

    print("Embedding test passages...")
    embeddings = model.encode(test_passages, normalize_embeddings=True)
    print(f"Embedded {len(embeddings)} passages, shape: {embeddings.shape}")

    print(f"\nInitializing local Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": test_passages[i]})
        for i in range(len(test_passages))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Upserted test points into Qdrant.")

    # Test query
    query = "query: कॉर्पोरेशन क्या है?"
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=3,
    )

    print(f"\n=== Query: {query} ===")
    for point in results.points:
        print(f"Score: {point.score:.4f} | Text: {point.payload['text']}")


if __name__ == "__main__":
    main()