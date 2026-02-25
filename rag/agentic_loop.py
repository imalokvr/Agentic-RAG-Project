"""Agentic RAG loop: up to 2-pass retrieve -> evaluate -> refine -> synthesize."""

from __future__ import annotations

from typing import List, Tuple

from models.schemas import (
    IterationTrace,
    RetrievalPlan,
    SynthesizedAnswer,
)
from rag.retriever import Retriever
from rag.evaluator import SufficiencyEvaluator
from rag.synthesizer import Synthesizer


class AgenticRAGLoop:
    """Orchestrates the 2-pass agentic retrieval loop."""

    def __init__(
        self,
        retriever: Retriever,
        evaluator: SufficiencyEvaluator,
        synthesizer: Synthesizer,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator
        self._synthesizer = synthesizer

    def run(
        self, plan: RetrievalPlan
    ) -> Tuple[SynthesizedAnswer, List[IterationTrace]]:
        iterations: List[IterationTrace] = []

        # ── Iteration 1 ───────────────────────────────────────────────
        chunks_1 = self._retriever.retrieve(plan.clean_query, plan.k)
        verdict_1 = self._evaluator.evaluate(plan.clean_query, chunks_1)

        iterations.append(
            IterationTrace(
                query=plan.clean_query,
                retrieved=chunks_1,
                evaluator=verdict_1,
            )
        )

        sources = ", ".join(dict.fromkeys(c.source for c in chunks_1))
        status = "sufficient -> synthesize" if verdict_1.sufficient else "insufficient -> iter 2"
        print(f"  [iter 1] {len(chunks_1)} chunks | {status} | confidence={verdict_1.confidence}")

        if verdict_1.sufficient:
            answer = self._synthesizer.synthesize(
                plan.clean_query, chunks_1, plan.notes
            )
            return answer, iterations

        # ── Iteration 2 (refined query) ───────────────────────────────
        refined_query = verdict_1.refined_query or plan.clean_query
        chunks_2 = self._retriever.retrieve(refined_query, plan.k)
        verdict_2 = self._evaluator.evaluate(refined_query, chunks_2)

        iterations.append(
            IterationTrace(
                query=refined_query,
                retrieved=chunks_2,
                evaluator=verdict_2,
            )
        )

        status2 = "sufficient" if verdict_2.sufficient else "insufficient (max reached)"
        print(f"  [iter 2] {len(chunks_2)} chunks | {status2} -> synthesize | confidence={verdict_2.confidence}")

        # Synthesize with whatever we have (note limitations if still insufficient)
        limitations = verdict_2.missing if not verdict_2.sufficient else None
        answer = self._synthesizer.synthesize(
            plan.clean_query, chunks_2, plan.notes, limitations
        )
        return answer, iterations
