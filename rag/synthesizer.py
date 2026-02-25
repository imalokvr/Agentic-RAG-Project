"""Answer synthesizer with [Cx] citations."""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_openai import AzureChatOpenAI

from models.schemas import RetrievedChunk, SynthesizedAnswer


class Synthesizer:
    """Generates a final answer from retrieved chunks, citing sources as [C1], [C2], etc."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def synthesize(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        notes: Optional[List[str]] = None,
        limitations: Optional[str] = None,
    ) -> SynthesizedAnswer:
        context_block = "\n\n".join(
            f"[{c.chunk_id}] (source: {c.source})\n{c.content}" for c in chunks
        )

        notes_block = ""
        if notes:
            notes_block = (
                "\nFormatting instructions from user: "
                + "; ".join(notes)
                + "\n"
            )

        limitations_block = ""
        if limitations:
            limitations_block = (
                f"\nNote: After two retrieval attempts, some information may still be incomplete. "
                f"Missing: {limitations}. Acknowledge any gaps honestly.\n"
            )

        prompt = f"""You are an HR policy assistant. Answer the user's question using ONLY the provided context chunks.

Rules:
- Cite every fact using the chunk ID in square brackets, e.g. [C1], [C2].
- If multiple chunks support a fact, cite all of them.
- If the context doesn't contain enough information, say so honestly.
- Do NOT invent information not present in the context.
{notes_block}{limitations_block}
Context Chunks:
{context_block}

User Question: {query}

Answer:"""

        resp = self._llm.invoke(prompt)
        answer_text = resp.content.strip()

        # Extract cited chunk IDs from the answer
        cited = sorted(set(re.findall(r"\[C\d+\]", answer_text)))

        return SynthesizedAnswer(
            answer=answer_text,
            citations_used=cited,
            limitations=limitations or "",
        )
