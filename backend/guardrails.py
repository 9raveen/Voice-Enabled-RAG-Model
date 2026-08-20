"""
Phase 6 — Guardrails for the RAG pipeline.
Pre-checks run before retrieval (cheap, fast rejection).
Post-checks run after generation (verify grounding).
"""

import re
from schemas import GeneratedAnswer, AnswerStatus, RetrievalResult

# --- Pre-checks: run before retrieval ---

MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 500

# Lightweight heuristic unsafe-content filter. Not a substitute for a
# real moderation model, but catches obvious cases cheaply before
# spending a retrieval + LLM call. Documented as a known limitation.
UNSAFE_PATTERNS = [
    r"\b(bomb|explosive|weapon)\s*(bana|kaise)\b",
    r"\bhack\s*(karo|kaise)\b",
    r"\bsuicide\b",
    r"\bself\s*harm\b",
]


def validate_input(query: str) -> tuple[bool, str | None]:
    """Returns (is_valid, rejection_reason)."""
    if not query or not query.strip():
        return False, "empty_input"

    stripped = query.strip()
    if len(stripped) < MIN_QUERY_LENGTH:
        return False, "too_short"
    if len(stripped) > MAX_QUERY_LENGTH:
        return False, "too_long"

    return True, None


def check_unsafe_content(query: str) -> tuple[bool, str | None]:
    """Returns (is_safe, matched_pattern_category)."""
    lowered = query.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return False, "unsafe_content_detected"
    return True, None


def rejection_response(query: str, reason: str) -> GeneratedAnswer:
    messages = {
        "empty_input": "कृपया एक वैध प्रश्न पूछें।",
        "too_short": "कृपया अपना प्रश्न अधिक स्पष्ट रूप से पूछें।",
        "too_long": "कृपया अपना प्रश्न संक्षेप में पूछें।",
        "unsafe_content_detected": "माफ़ कीजिए, मैं इस तरह के प्रश्न का उत्तर नहीं दे सकता।",
    }
    return GeneratedAnswer(
        status=AnswerStatus.NO_ANSWER,
        answer=messages.get(reason, "क्षमा करें, इस प्रश्न को संसाधित नहीं किया जा सका।"),
        source_chunk_ids=[],
        error_message=f"Rejected at pre-check: {reason}",
    )

# --- Post-checks: run after generation ---

from sentence_transformers import CrossEncoder

_reranker = None

GROUNDING_THRESHOLD = 0.3  # cross-encoder score scale differs from cosine sim — will calibrate empirically


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            model_kwargs={"torch_dtype": "float16"},
        )
    return _reranker


def check_grounding(answer_text: str, retrieval: RetrievalResult) -> tuple[bool, float]:
    """Returns (is_grounded, score). Uses the cross-encoder to directly
    score how well the answer is supported by the retrieved context,
    rather than comparing independent embeddings."""
    if not answer_text or not retrieval.chunks:
        return False, 0.0

    reranker = _get_reranker()

    context_text = " ".join(c.text for c in retrieval.chunks)
    score = reranker.predict([[context_text, answer_text]])[0]

    is_grounded = score >= GROUNDING_THRESHOLD
    return is_grounded, float(score)