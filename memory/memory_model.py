"""Conversation memory: rolling history, summary, and extracted facts."""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI

from config.settings import MAX_HISTORY_TURNS
from models.schemas import MemoryState


class ConversationMemory:
    """Maintains a sliding window of history plus a running LLM summary."""

    def __init__(self) -> None:
        self._state = MemoryState()

    def get_state(self) -> MemoryState:
        return self._state.model_copy(deep=True)

    def add_turn(self, role: str, content: str) -> None:
        self._state.history.append({"role": role, "content": content})
        # trim oldest turns when we exceed the window
        if len(self._state.history) > MAX_HISTORY_TURNS * 2:
            self._state.history = self._state.history[-(MAX_HISTORY_TURNS * 2) :]

    def update_summary(self, llm: AzureChatOpenAI) -> None:
        """Ask the LLM to produce a concise summary of conversation so far."""
        if not self._state.history:
            return

        recent = self._state.history[-6:]  # last 3 user+assistant pairs
        turns_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in recent
        )

        prompt = (
            "Summarize the following conversation in 2-3 sentences. "
            "Focus on the topics discussed and any user preferences.\n\n"
            f"Previous summary: {self._state.summary or '(none)'}\n\n"
            f"Recent turns:\n{turns_text}\n\nSummary:"
        )
        resp = llm.invoke(prompt)
        self._state.summary = resp.content.strip()

    def extract_facts(self, llm: AzureChatOpenAI, message: str) -> None:
        """Extract user preferences or facts from a message."""
        prompt = (
            "Extract any user preferences or explicit instructions from this message. "
            "Return a JSON list of short strings, or an empty list if none found.\n\n"
            f"Message: {message}\n\nFacts:"
        )
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        # simple parse: look for bracketed list items
        import json
        try:
            facts = json.loads(text)
            if isinstance(facts, list):
                for f in facts:
                    if isinstance(f, str) and f not in self._state.facts:
                        self._state.facts.append(f)
        except json.JSONDecodeError:
            pass
