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


LANGUAGE_TO_STT_CODE = {
    "hi": "hi-IN",
    "en": "en-IN",
}


def run_pipeline_from_audio(audio_path: str, language: str = "hi") -> PipelineResponse:
    """Voice entry point: STT -> text pipeline."""
    timings = {}
    stt_code = LANGUAGE_TO_STT_CODE.get(language, "hi-IN")
    transcript, stt_ms, stt_error = transcribe(audio_path, language_code=stt_code)
    timings["stt_ms"] = stt_ms

    if stt_error or not transcript:
        stt_err_msg = (
            "Sorry, there was an issue understanding the audio."
            if language == "en"
            else "माफ़ कीजिए, आवाज़ को समझने में समस्या हुई।"
        )
        return PipelineResponse(
            query="",
            status=AnswerStatus.ERROR,
            answer=stt_err_msg,
            sources=[],
            latency_ms=timings,
        )

    result = run_pipeline(transcript, language=language)
    # merge STT timing into the text pipeline's timing breakdown
    result.latency_ms = {**timings, **result.latency_ms}
    result.latency_ms["total_ms"] = stt_ms + result.latency_ms.get("total_ms", 0)
    return result


def run_pipeline(query: str, language: str = "hi") -> PipelineResponse:
    timings = {}
    t_start = time.time()

    # 1. Input validation
    t0 = time.time()
    is_valid, reason = validate_input(query)
    timings["validation_ms"] = (time.time() - t0) * 1000
    if not is_valid:
        rejected = rejection_response(query, reason, language=language)
        return PipelineResponse(
            query=query, status=rejected.status, answer=rejected.answer,
            sources=[], latency_ms=timings,
        )

    # 2. Unsafe content check
    t0 = time.time()
    is_safe, safety_reason = check_unsafe_content(query)
    timings["safety_check_ms"] = (time.time() - t0) * 1000
    if not is_safe:
        rejected = rejection_response(query, safety_reason, language=language)
        return PipelineResponse(
            query=query, status=rejected.status, answer=rejected.answer,
            sources=[], latency_ms=timings,
        )

    # 3. Retrieval (with confidence gating and failure recovery built in)
    t0 = time.time()
    retrieval_result = retrieve(query, language=language)
    timings["retrieval_ms"] = (time.time() - t0) * 1000

    # 4. Generation (handles low-confidence retrieval internally too)
    t0 = time.time()
    generated = generate_answer(retrieval_result, language=language)
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
            generated.answer = (
                "Sorry, I could not find a reliable answer to this question in the provided context."
                if language == "en"
                else "माफ़ कीजिए, मुझे इस सवाल का भरोसेमंद जवाब नहीं मिला।"
            )

    timings["total_ms"] = (time.time() - t_start) * 1000

    sources = [c.text for c in retrieval_result.chunks if c.chunk_id in generated.source_chunk_ids]

    return PipelineResponse(
        query=query,
        status=generated.status,
        answer=generated.answer,
        sources=sources,
        latency_ms=timings,
    )