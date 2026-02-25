"""ChatOrchestrator: wires all components and handles the full query pipeline."""

from __future__ import annotations

from config.settings import VECTOR_STORE_DIR, TRACES_DIR, make_llm, make_embeddings
from ingestion.embedder import load_index
from memory.memory_model import ConversationMemory
from agents.front_agent import FrontAgent
from rag.retriever import Retriever
from rag.evaluator import SufficiencyEvaluator
from rag.synthesizer import Synthesizer
from rag.agentic_loop import AgenticRAGLoop
from orchestrator.tracer import QueryTracer


class ChatOrchestrator:
    """End-to-end pipeline: user message → answer (with tracing + memory)."""

    def __init__(self) -> None:
        print("[orchestrator] Initializing components...")
        self._llm = make_llm()
        embeddings = make_embeddings()
        faiss_index = load_index(VECTOR_STORE_DIR, embeddings)

        self._memory = ConversationMemory()
        self._front_agent = FrontAgent(self._llm)
        retriever = Retriever(faiss_index)
        evaluator = SufficiencyEvaluator(self._llm)
        synthesizer = Synthesizer(self._llm)
        self._rag_loop = AgenticRAGLoop(retriever, evaluator, synthesizer)

        print("[orchestrator] Ready.\n")

    def handle_query(self, user_message: str) -> str:
        tracer = QueryTracer()

        # 1. Start trace
        state = self._memory.get_state()
        tracer.start_query(user_message, state.summary)

        # 2. Front Agent planning
        plan = self._front_agent.plan(user_message, state)
        tracer.set_plan(plan)
        notes_str = f"  notes={plan.notes}" if plan.notes else ""
        print(f"  [plan] {plan.clean_query!r}  k={plan.k}{notes_str}")

        # 3. Agentic RAG loop
        answer, iterations = self._rag_loop.run(plan)

        # 4. Record iterations in trace
        for i, it in enumerate(iterations, start=1):
            tracer.set_iteration(i, it)

        # 5. Record answer
        tracer.set_answer(answer)

        # 6. Save trace
        tracer.save(TRACES_DIR)

        # 7. Update memory
        self._memory.add_turn("user", user_message)
        self._memory.add_turn("assistant", answer.answer)
        self._memory.update_summary(self._llm)

        return answer.answer
