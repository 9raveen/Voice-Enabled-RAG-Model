"""
Phase 6, Step 3 — Full orchestrated pipeline.
validate -> safety check -> retrieve -> confidence gate -> generate
-> grounding check -> final structured response.

This is the harness your brief describes end to end. STT and the
API layer (Phase 9/10) will call run_pipeline() as the single entry point.
"""

import time
from schemas import PipelineResponse, AnswerStatus
from guardrails import validate_input, check_unsafe_content, rejection_response, check_grounding
from retrieval import retrieve
from generation import generate_answer
from stt import transcribe


def run_pipeline_from_audio(audio_path: str) -> PipelineResponse:
    """Voice entry point: STT -> text pipeline."""
    timings = {}
    transcript, stt_ms, stt_error = transcribe(audio_path)
    timings["stt_ms"] = stt_ms

    if stt_error or not transcript:
        return PipelineResponse(
            query="",
            status=AnswerStatus.ERROR,
            answer="माफ़ कीजिए, आवाज़ को समझने में समस्या हुई।",
            sources=[],
            latency_ms=timings,
        )

    result = run_pipeline(transcript)
    # merge STT timing into the text pipeline's timing breakdown
    result.latency_ms = {**timings, **result.latency_ms}
    result.latency_ms["total_ms"] = stt_ms + result.latency_ms.get("total_ms", 0)
    return result


def run_pipeline(query: str) -> PipelineResponse:
    timings = {}
    t_start = time.time()

    # 1. Input validation
    t0 = time.time()
    is_valid, reason = validate_input(query)
    timings["validation_ms"] = (time.time() - t0) * 1000
    if not is_valid:
        rejected = rejection_response(query, reason)
        return PipelineResponse(
            query=query, status=rejected.status, answer=rejected.answer,
            sources=[], latency_ms=timings,
        )

    # 2. Unsafe content check
    t0 = time.time()
    is_safe, safety_reason = check_unsafe_content(query)
    timings["safety_check_ms"] = (time.time() - t0) * 1000
    if not is_safe:
        rejected = rejection_response(query, safety_reason)
        return PipelineResponse(
            query=query, status=rejected.status, answer=rejected.answer,
            sources=[], latency_ms=timings,
        )

    # 3. Retrieval (with confidence gating and failure recovery built in)
    t0 = time.time()
    retrieval_result = retrieve(query)
    timings["retrieval_ms"] = (time.time() - t0) * 1000

    # 4. Generation (handles low-confidence retrieval internally too)
    t0 = time.time()
    generated = generate_answer(retrieval_result)
    timings["generation_ms"] = (time.time() - t0) * 1000

    # 5. Grounding check - only meaningful if we actually answered
    grounded = True
    grounding_score = None
    if generated.status == AnswerStatus.ANSWERED:
        t0 = time.time()
        grounded, grounding_score = check_grounding(generated.answer, retrieval_result)
        timings["grounding_check_ms"] = (time.time() - t0) * 1000
        
        if not grounded:
            # Grounding check failed post-hoc - override to a safe refusal
            # rather than returning a potentially hallucinated answer.
            generated.status = AnswerStatus.NO_ANSWER
            generated.answer = "माफ़ कीजिए, मुझे इस सवाल का भरोसेमंद जवाब नहीं मिला।"

    timings["total_ms"] = (time.time() - t_start) * 1000

    sources = [c.text for c in retrieval_result.chunks if c.chunk_id in generated.source_chunk_ids]

    return PipelineResponse(
        query=query,
        status=generated.status,
        answer=generated.answer,
        sources=sources,
        latency_ms=timings,
    )