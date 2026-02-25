# Agentic RAG Chat Assistant

Agentic RAG chat assistant that uses a Front Agent for memory-driven query planning and a 2-pass retrieval loop with sufficiency evaluation, chunk-level [Cx] citations, and per-query JSON trace output.

---

## Features

- **N-Turn Conversation Memory** — rolling history, LLM-generated summaries, and pronoun resolution across turns
- **Front Agent** — rewrites user queries into searchable form using conversation context
- **2-Pass Agentic RAG Loop** — retrieves chunks, evaluates sufficiency, refines query if needed, then synthesizes
- **Chunk Citations** — every fact in the answer is cited as [C1], [C2], etc.
- **Per-Query Tracing** — full trace.json saved for each query (plan, iterations, evaluator verdicts, final answer)
- **Streamlit UI** — browser-based chat with a trace viewer sidebar (stretch goal)

---

## Project Structure

```
agentic_rag_project/
├── config/settings.py          # Env, paths, LLM/embedding factories
├── models/schemas.py           # Pydantic data contracts (8 models)
│
├── ingestion/                  # Document Processing
│   ├── loader.py               #   .docx -> LangChain Documents
│   ├── chunker.py              #   Documents -> Semantic Chunks
│   ├── embedder.py             #   Chunks -> FAISS Vector Index
│   └── ingest_pipeline.py      #   Orchestrates load -> chunk -> embed -> persist
│
├── memory/                     # Conversation State
│   └── memory_model.py         #   History + Summary + Facts
│
├── agents/                     # Query Planning
│   └── front_agent.py          #   Message + Memory -> RetrievalPlan
│
├── rag/                        # Agentic Retrieval
│   ├── retriever.py            #   FAISS search -> RetrievedChunks
│   ├── evaluator.py            #   Sufficiency check -> Verdict
│   ├── synthesizer.py          #   Chunks -> Answer with [Cx] citations
│   └── agentic_loop.py         #   2-pass: retrieve -> evaluate -> refine -> synthesize
│
├── orchestrator/               # Glue + Observability
│   ├── orchestrator.py         #   Wires all components
│   └── tracer.py               #   Saves per-query trace.json
│
├── app/                        # User Interface
│   ├── run_chat.py             #   CLI entry point
│   └── app.py                  #   Streamlit UI
│
├── docs/                       # 10 HR policy .docx files
├── vector_store/               # Persisted FAISS index (generated)
└── traces/                     # Per-query trace.json output (generated)
```

---

## How It Works

```
User Input
    |
    v
Orchestrator
    |-- Memory.get_state() -> conversation history + summary
    |-- FrontAgent.plan(msg, memory) -> RetrievalPlan (resolved query, k, notes)
    |-- AgenticRAGLoop.run(plan)
    |       |-- Retriever.retrieve(query, k) -> chunks with [C1], [C2], ...
    |       |-- Evaluator.evaluate(query, chunks) -> sufficient?
    |       |       |-- YES -> Synthesizer
    |       |       |-- NO  -> Retriever.retrieve(refined_query, k) -> Synthesizer
    |       |-- Synthesizer.synthesize(query, chunks, notes) -> answer with citations
    |-- Tracer.save() -> trace.json
    |-- Memory.add_turn() + update_summary()
    v
Answer displayed to user
```

---

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI access (GPT-4o-mini + text-embedding-3-small)

### Install

```bash
cd C:\repo\training\agentic_rag_project
pip install -r requirements.txt
```

### Configure

The `.env` file contains Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini-alok
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small-alok
```

---

## Usage

### Step 1: Run Ingestion (one-time)

```bash
python -m ingestion.ingest_pipeline
```

Loads 10 HR policy `.docx` files, chunks them semantically, and builds a FAISS vector index.

### Step 2: Run CLI Chat

```bash
python -m app.run_chat
```

### Step 3 (Optional): Run Streamlit UI

```bash
python -m streamlit run app/app.py
```

Opens at http://localhost:8501 with chat interface and trace viewer sidebar.

---

## Test Prompts

Run these 3 prompts in sequence to verify all features:

| # | Prompt | Tests |
|---|--------|-------|
| 1 | "What is the sick leave carry forward limit?" | 1-pass retrieval, [Cx] citations |
| 2 | "Explain sick leave carry forward with an example." | 2-pass retrieval (evaluator triggers refinement) |
| 3 | "Now summarize that in 2 lines." | Memory (resolves "that"), formatting notes |

### Expected Output

```
You: What is the sick leave carry forward limit?
  [plan] 'What is the sick leave carry forward limit in the HR policy?'  k=8
  [iter 1] 8 chunks | sufficient -> synthesize | confidence=1.0
  [tracer] Saved -> 20260225T..._trace.json