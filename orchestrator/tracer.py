"""Per-query trace builder and JSON persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models.schemas import (
    IterationTrace,
    QueryTrace,
    RetrievalPlan,
    SynthesizedAnswer,
)


class QueryTracer:
    """Builds a QueryTrace incrementally and writes it to disk."""

    def __init__(self) -> None:
        self._trace: QueryTrace | None = None

    def start_query(self, user_message: str, memory_summary: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        hex4 = uuid.uuid4().hex[:4]
        run_id = f"{ts}_{hex4}"
        self._trace = QueryTrace(
            run_id=run_id,
            user_message=user_message,
            memory_summary=memory_summary,
        )

    def set_plan(self, plan: RetrievalPlan) -> None:
        if self._trace:
            self._trace.retrieval_plan = plan

    def set_iteration(self, num: int, trace: IterationTrace) -> None:
        if not self._trace:
            return
        if num == 1:
            self._trace.iteration_1 = trace
        elif num == 2:
            self._trace.iteration_2 = trace

    def set_answer(self, answer: SynthesizedAnswer) -> None:
        if self._trace:
            self._trace.final_answer = answer.answer
            self._trace.citations_used = answer.citations_used

    def save(self, output_dir: Path) -> Path:
        if not self._trace:
            raise RuntimeError("No active trace to save.")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._trace.run_id}_trace.json"
        path.write_text(
            self._trace.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"  [tracer] Saved -> {path.name}")
        return path
