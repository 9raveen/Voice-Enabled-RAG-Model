"""
Phase 5, Step 1 — Structured I/O schemas for the RAG pipeline.
Every stage of the harness validates against one of these, so
failures are caught explicitly rather than surfacing as vague errors.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_passage_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_score: float
    confident: bool  # set by a threshold check — see guardrails in Phase 6


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"           # retrieval too weak / off-topic
    ERROR = "error"                    # something failed upstream


class GeneratedAnswer(BaseModel):
    status: AnswerStatus
    answer: Optional[str] = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    reasoning_tokens: Optional[int] = None
    model_used: Optional[str] = None
    error_message: Optional[str] = None


class PipelineResponse(BaseModel):
    """Final response returned to the caller (API/frontend)."""
    query: str
    status: AnswerStatus
    answer: Optional[str] = None
    sources: list[str] = Field(default_factory=list)   # passage texts used
    latency_ms: dict = Field(default_factory=dict)       # per-stage timing