"""
Phase 9 (revised) — FastAPI backend exposing the RAG pipeline
as a real HTTP API for the website frontend to call.
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import run_pipeline, run_pipeline_from_audio

app = FastAPI(title="Voice RAG API")

# Allow the frontend (served separately) to call this API.
# Tightened to specific origins once we know the deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    query: str
    language: str = "hi"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/text")
def ask_text(payload: TextQuery):
    result = run_pipeline(payload.query, language=payload.language)
    return result.model_dump()

from fastapi import Form

@app.post("/ask/voice")
async def ask_voice(audio: UploadFile = File(...), language: str = Form("hi")):
    # Save uploaded audio to a temp file (Sarvam SDK expects a file path/object)
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_pipeline_from_audio(tmp_path, language=language)
        return result.model_dump()
    finally:
        os.unlink(tmp_path)