"""
Phase 9, Step 2 — STT stage of the harness. Wraps Sarvam with the
same fail-safe pattern as retrieval.py and generation.py.
"""

import os
import time
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

_client = None
MODEL = "saaras:v3"


def _get_client():
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
    return _client


def transcribe(audio_path: str, language_code: str = "hi-IN") -> tuple[str | None, float, str | None]:
    """Returns (transcript, latency_ms, error_message)."""
    client = _get_client()
    start = time.time()
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                language_code=language_code,
                model=MODEL,
            )
        elapsed_ms = (time.time() - start) * 1000
        return response.transcript, elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, str(e)