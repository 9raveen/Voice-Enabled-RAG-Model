"""
Phase 6, Step 2 — Test the grounding check against known cases:
one grounded answer, one deliberately fabricated/ungrounded answer.
"""

from schemas import RetrievalResult, RetrievedChunk
from guardrails import check_grounding

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

grounded_answer = "कॉर्पोरेशन एक कंपनी है जो कानूनी रूप से अपने मालिकों से अलग होती है।"
fabricated_answer = "ताजमहल भारत में स्थित एक ऐतिहासिक स्मारक है जो सफेद संगमरमर से बना है।"

for label, answer in [("Grounded answer", grounded_answer), ("Fabricated/unrelated answer", fabricated_answer)]:
    is_grounded, score = check_grounding(answer, retrieval)
    print(f"{label}: grounded={is_grounded}, similarity={score:.4f}")