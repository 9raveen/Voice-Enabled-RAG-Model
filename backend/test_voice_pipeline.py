"""
Phase 9, Step 2 — Full voice-to-answer pipeline test using real audio.
"""

from pipeline import run_pipeline_from_audio

AUDIO_PATH = "../data/test_audio.wav"

result = run_pipeline_from_audio(AUDIO_PATH)

print(f"Status: {result.status}")
print(f"Answer: {result.answer}")
print(f"Latency breakdown: {result.latency_ms}")