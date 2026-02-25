"""Front Agent: takes user message + memory state → RetrievalPlan."""

from __future__ import annotations

import json
import re

from langchain_openai import AzureChatOpenAI

from config.settings import DEFAULT_TOP_K
from models.schemas import MemoryState, RetrievalPlan


class FrontAgent:
    """Rewrites user queries into searchable retrieval plans using memory context."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def plan(self, user_message: str, memory: MemoryState) -> RetrievalPlan:
        # Build context from memory
        summary_block = (
            f"Conversation summary: {memory.summary}" if memory.summary else ""
        )
        recent_turns = memory.history[-6:]  # last 3 pairs
        history_block = "\n".join(
            f"  {t['role'].upper()}: {t['content']}" for t in recent_turns
        )
        facts_block = (
            "User preferences: " + "; ".join(memory.facts) if memory.facts else ""
        )

        prompt = f"""You are a query-planning agent for an HR policy RAG system.

Given the conversation context and the user's latest message, produce a JSON object with:
- "clean_query": a self-contained, searchable query that resolves all pronouns and references using conversation history. This must be a standalone question that a search engine could answer without any conversation context.
- "k": number of chunks to retrieve (default {DEFAULT_TOP_K}, increase for broad topics)
- "notes": list of formatting/style instructions extracted from the user message (e.g. ["keep to 2 lines", "give an example"])

IMPORTANT: If the user says "that", "it", "this", or similar pronouns, you MUST resolve them to the actual topic from conversation history.

{summary_block}

Recent conversation:
{history_block}

{facts_block}

Current user message: {user_message}

Respond ONLY with a valid JSON object. No markdown, no explanation."""

        resp = self._llm.invoke(prompt)
        text = resp.content.strip()

        # Try direct JSON parse first
        try:
            data = json.loads(text)
            return RetrievalPlan(
                clean_query=data.get("clean_query", user_message),
                k=data.get("k", DEFAULT_TOP_K),
                notes=data.get("notes", []),
            )
        except (json.JSONDecodeError, KeyError):
            pass

        # Regex fallback: extract fields from malformed JSON
        cq_match = re.search(r'"clean_query"\s*:\s*"([^"]+)"', text)
        k_match = re.search(r'"k"\s*:\s*(\d+)', text)

        return RetrievalPlan(
            clean_query=cq_match.group(1) if cq_match else user_message,
            k=int(k_match.group(1)) if k_match else DEFAULT_TOP_K,
            notes=[],
        )
