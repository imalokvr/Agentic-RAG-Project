"""Pydantic data contracts shared across all components."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Memory ─────────────────────────────────────────────────────────────
class MemoryState(BaseModel):
    history: List[Dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    facts: List[str] = Field(default_factory=list)


# ── Front Agent → RAG Loop ─────────────────────────────────────────────
class RetrievalPlan(BaseModel):
    clean_query: str
    k: int = 8
    notes: List[str] = Field(default_factory=list)


# ── Retriever output ───────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    chunk_id: str          # e.g. "C1", "C2"
    content: str
    source: str = ""
    page: int = 0
    score: float = 0.0


# ── Evaluator output ──────────────────────────────────────────────────
class SufficiencyVerdict(BaseModel):
    sufficient: bool
    missing: str = ""
    refined_query: str = ""
    confidence: float = 0.0


# ── Synthesizer output ────────────────────────────────────────────────
class SynthesizedAnswer(BaseModel):
    answer: str
    citations_used: List[str] = Field(default_factory=list)
    limitations: str = ""


# ── Tracing ────────────────────────────────────────────────────────────
class IterationTrace(BaseModel):
    query: str
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    evaluator: Optional[SufficiencyVerdict] = None


class QueryTrace(BaseModel):
    run_id: str
    user_message: str
    memory_summary: str = ""
    retrieval_plan: Optional[RetrievalPlan] = None
    iteration_1: Optional[IterationTrace] = None
    iteration_2: Optional[IterationTrace] = None
    final_answer: str = ""
    citations_used: List[str] = Field(default_factory=list)
