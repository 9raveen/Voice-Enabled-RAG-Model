"""
Phase 9, Step 1 — Sanity check: Sarvam STT on a real audio file.
"""

import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

AUDIO_PATH = "data/test_audio.wav"

with open(AUDIO_PATH, "rb") as audio_file:
    response = client.speech_to_text.transcribe(
        file=audio_file,
        language_code="hi-IN",
        model="saaras:v3",
    )

print(f"Transcript: {response.transcript}")