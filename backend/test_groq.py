"""
Phase 5, Step 1 — Sanity check: confirm the Groq model works and
returns sensible Hindi output before building the full harness.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-120b"

response = client.chat.completions.create(
    model=MODEL,
    reasoning_effort="low",
    messages=[
        {
            "role": "system",
            "content": "You answer questions in Hindi based only on the provided context. If the context doesn't contain the answer, say so in Hindi.",
        },
        {
            "role": "user",
            "content": (
                "संदर्भ: कॉर्पोरेशन एक कंपनी होती है जो कानूनी रूप से अपने मालिकों से अलग होती है।\n\n"
                "प्रश्न: कॉर्पोरेशन क्या है?"
            ),
        },
    ],
)

print(response.choices[0].message.content)
print(f"\nModel used: {response.model}")
print(f"Tokens: {response.usage}")