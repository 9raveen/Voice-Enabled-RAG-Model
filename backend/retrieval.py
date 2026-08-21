"""
Phase 5, Step 3 — Retrieval stage of the harness.
Wraps Qdrant search behind a clean function that returns a validated
RetrievalResult, including the confidence check that gates generation.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from schemas import RetrievalResult, RetrievedChunk

QDRANT_URL = "http://localhost:6333"
COLLECTIONS = {
    "hi": "voice_rag_corpus",
    "en": "voice_rag_corpus_en",
}
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.75  # tuned from Phase 4 score distributions

_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cuda")
        _model.half()
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def retrieve(query: str, language: str = "hi", top_k: int = TOP_K) -> RetrievalResult:
    collection_name = COLLECTIONS.get(language, COLLECTIONS["hi"])
    try:
        model = _get_model()
        client = _get_client()

        query_vec = model.encode(["query: " + query], normalize_embeddings=True)[0]

        results = client.query_points(
            collection_name=collection_name,
            query=query_vec.tolist(),
            limit=top_k,
        )
    except Exception as e:
        # Retrieval failure (Qdrant down, embedding error, etc.) - fail
        # safe with an empty, non-confident result rather than crashing.
        return RetrievalResult(query=query, chunks=[], top_score=0.0, confident=False)

    chunks = [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            source_passage_id=p.payload["source_passage_id"],
            text=p.payload["text"],
            score=p.score,
            metadata=p.payload.get("metadata", {}),
        )
        for p in results.points
    ]

    top_score = chunks[0].score if chunks else 0.0
    confident = top_score >= CONFIDENCE_THRESHOLD

    return RetrievalResult(
        query=query,
        chunks=chunks,
        top_score=top_score,
        confident=confident,
    )