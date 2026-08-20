"""
Phase 5, Step 2 — Test the generation harness with a real retrieval
result (built manually here from known-good data, before we wire in
live Qdrant retrieval in the next step).
"""

from schemas import RetrievalResult, RetrievedChunk
from generation import generate_answer

# Manually constructed retrieval result, using the same test passage
# we validated back in Phase 3, so we know the correct answer.
retrieval = RetrievalResult(
    query="कॉर्पोरेशन क्या है?",
    chunks=[
        RetrievedChunk(
            chunk_id="test001",
            source_passage_id="test001",
            text="कॉर्पोरेशन एक कंपनी होती है जो कानूनी रूप से अपने मालिकों से अलग होती है।",
            score=0.88,
        )
    ],
    top_score=0.88,
    confident=True,
)

result = generate_answer(retrieval)

print(f"Status: {result.status}")
print(f"Answer: {result.answer}")
print(f"Source chunks: {result.source_chunk_ids}")
print(f"Reasoning tokens: {result.reasoning_tokens}")
print(f"Model: {result.model_used}")
if result.error_message:
    print(f"Error: {result.error_message}")