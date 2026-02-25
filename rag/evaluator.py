"""Sufficiency evaluator: decides if retrieved chunks answer the query."""

from __future__ import annotations

import json
import re

from typing import List

from langchain_openai import AzureChatOpenAI

from models.schemas import RetrievedChunk, SufficiencyVerdict


class SufficiencyEvaluator:
    """LLM-based judge that decides whether retrieved context is sufficient."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def evaluate(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> SufficiencyVerdict:
        context_block = "\n\n".join(
            f"[{c.chunk_id}] (source: {c.source})\n{c.content}" for c in chunks
        )

        prompt = f"""You are a retrieval quality evaluator for an HR policy RAG system.

Given a user query and retrieved document chunks, determine if the chunks contain SUFFICIENT information to fully answer the query.

Be strict: if the query asks for an explanation with examples but the chunks only contain policy statements without examples, mark as insufficient. If the query asks for a specific detail and the chunks discuss the topic generally but don't include that detail, mark as insufficient.

User Query: {query}

Retrieved Chunks:
{context_block}

Respond with a JSON object:
{{
  "sufficient": true/false,
  "missing": "what information is missing (empty string if sufficient)",
  "refined_query": "a better search query to find the missing info (empty string if sufficient)",
  "confidence": 0.0 to 1.0
}}

Respond ONLY with valid JSON. No markdown, no explanation."""

        resp = self._llm.invoke(prompt)
        text = resp.content.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
            return SufficiencyVerdict(
                sufficient=bool(data.get("sufficient", True)),
                missing=data.get("missing", ""),
                refined_query=data.get("refined_query", ""),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback: assume sufficient to avoid infinite loops
            return SufficiencyVerdict(
                sufficient=True,
                missing="",
                refined_query="",
                confidence=0.5,
            )
