"""
Phase 5, Step 2 — Generation stage of the harness.
Takes a RetrievalResult, builds a grounded prompt, calls Groq with
retries/timeout, and returns a validated GeneratedAnswer.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq, APITimeoutError, APIError
from schemas import RetrievalResult, GeneratedAnswer, AnswerStatus

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 2
TIMEOUT_SECONDS = 8
CONFIDENCE_THRESHOLD = 0.5  # tune later once we have more query data

SYSTEM_PROMPTS = {
    "hi": (
        "आप एक सहायक हैं जो केवल दिए गए संदर्भ के आधार पर हिंदी में जवाब देते हैं। "
        "यदि संदर्भ में उत्तर नहीं है, तो स्पष्ट रूप से कहें कि जानकारी उपलब्ध नहीं है। "
        "संदर्भ के बाहर की जानकारी का उपयोग न करें।"
    ),
    "en": (
        "You are an assistant that answers strictly in English based ONLY on the provided context. "
        "Do not use any external knowledge. If the context does not contain the answer, "
        "clearly state in English that the information is not available in the context. "
        "Always respond in clear, grammatically correct English."
    ),
}


def build_prompt(retrieval: RetrievalResult, language: str = "hi") -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieval.chunks, start=1):
        context_blocks.append(f"[{i}] {chunk.text}")
    context = "\n\n".join(context_blocks)
    if language == "en":
        return (
            f"Context:\n{context}\n\n"
            f"Question: {retrieval.query}\n\n"
            f"Instruction: Answer the question in English using only the context above."
        )
    return (
        f"संदर्भ:\n{context}\n\n"
        f"प्रश्न: {retrieval.query}\n\n"
        f"निर्देश: केवल ऊपर दिए गए संदर्भ के आधार पर हिंदी में उत्तर दें।"
    )

def generate_answer(retrieval: RetrievalResult, language: str = "hi") -> GeneratedAnswer:
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["hi"])
    # Guardrail stub: if retrieval wasn't confident, don't even call the LLM.
    # Full guardrail logic comes in Phase 6 — this is the harness's hook point.
    if not retrieval.confident or not retrieval.chunks:
        no_ans = (
            "Sorry, I could not find a reliable answer to this question in the provided context."
            if language == "en"
            else "माफ़ कीजिए, मुझे इस सवाल का भरोसेमंद जवाब नहीं मिला।"
        )
        return GeneratedAnswer(
            status=AnswerStatus.NO_ANSWER,
            answer=no_ans,
            source_chunk_ids=[],
        )

    prompt = build_prompt(retrieval, language=language)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # initial try + retries
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                reasoning_effort="low",
                timeout=TIMEOUT_SECONDS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            elapsed = time.time() - start

            answer_text = response.choices[0].message.content

            return GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                answer=answer_text,
                source_chunk_ids=[c.chunk_id for c in retrieval.chunks],
                reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens
                    if response.usage.completion_tokens_details else None,
                model_used=response.model,
            )

        except APITimeoutError as e:
            last_error = f"Timeout on attempt {attempt}: {e}"
        except APIError as e:
            last_error = f"API error on attempt {attempt}: {e}"
        except Exception as e:
            last_error = f"Unexpected error on attempt {attempt}: {e}"

        if attempt <= MAX_RETRIES:
            time.sleep(0.5 * attempt)  # simple backoff

    # All retries exhausted
    return GeneratedAnswer(
        status=AnswerStatus.ERROR,
        error_message=last_error,
    )