# Agentic RAG Chat Assistant — Architecture & Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (CLI / Streamlit)                       │
│                                                                     │
│   Prompt 1: "What is the sick leave carry forward limit?"           │
│   Prompt 2: "Explain sick leave carry forward with an example."     │
│   Prompt 3: "Now summarize that in 2 lines."                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (orchestrator.py)                   │
│                                                                      │
│  Wires all components together. For each user message:               │
│                                                                      │
│  1. Read memory state                                                │
│  2. Call Front Agent for query planning                              │
│  3. Run Agentic RAG Loop                                            │
│  4. Save trace to JSON                                               │
│  5. Update conversation memory                                       │
│  6. Return answer to user                                            │
└──────┬───────────┬────────────────┬──────────────┬───────────────────┘
       │           │                │              │
       v           v                v              v
   ┌────────┐ ┌──────────┐  ┌────────────┐  ┌──────────┐
   │ MEMORY │ │  FRONT   │  │  AGENTIC   │  │  TRACER  │
   │        │ │  AGENT   │  │  RAG LOOP  │  │          │
   └────────┘ └──────────┘  └────────────┘  └──────────┘
```

---

## Two Core Components

### Component 1: Front Agent (Memory + Query Planning)

```
                    ┌───────────────────────────────────┐
                    │      CONVERSATION MEMORY           │
                    │                                   │
                    │  history: [                        │
                    │    {user: "What is sick leave..."}│
                    │    {assistant: "The limit is..."}  │
                    │  ]                                 │
                    │  summary: "User asked about..."    │
                    │  facts: ["keep answers brief"]     │
                    └───────────────┬───────────────────┘
                                    │
                 User says:         │  Memory State
        "Now summarize that         │  (history + summary + facts)
          in 2 lines."             │
                │                   │
                v                   v
        ┌───────────────────────────────────────────┐
        │              FRONT AGENT                   │
        │                                           │
        │  INPUT:                                    │
        │    - user message: "Now summarize that     │
        │                     in 2 lines."           │
        │    - memory state (summary, history, facts)│
        │                                           │
        │  LLM TASK:                                 │
        │    - Resolve pronouns ("that" -> sick      │
        │      leave carry forward from history)     │
        │    - Rewrite into searchable query         │
        │    - Extract formatting notes              │
        │                                           │
        │  OUTPUT (RetrievalPlan):                   │
        │    clean_query: "Summarize sick leave       │
        │                  carry forward policy"     │
        │    k: 8                                    │
        │    notes: ["keep to 2 lines"]              │
        └───────────────────────────────────────────┘
```

**Why is this needed?**
- User says "that", "it", "this" — the RAG system can't search for pronouns
- Memory lets the agent resolve references to actual topics
- Notes capture formatting preferences (e.g., "2 lines", "give example")

---

### Component 2: Agentic RAG Loop (2-Pass Retrieval)

```
        RetrievalPlan
        (clean_query, k, notes)
                │
                v
┌───────────────────────────────────────────────────────────────┐
│                    AGENTIC RAG LOOP                            │
│                                                               │
│  ┌─────────────── ITERATION 1 ──────────────────┐            │
│  │                                               │            │
│  │  ┌───────────┐    query: "sick leave          │            │
│  │  │ RETRIEVER │     carry forward limit"        │            │
│  │  │           │                                │            │
│  │  │ FAISS     │──> [C1] Leave policy sec 4.2   │            │
│  │  │ Vector DB │    [C2] Leave policy sec 3.1   │            │
│  │  │           │    [C3] General work policy     │            │
│  │  │ top-k     │    ...                         │            │
│  │  │ similarity│                                │            │
│  │  └───────────┘                                │            │
│  │        │                                      │            │
│  │        v                                      │            │
│  │  ┌─────────────┐                              │            │
│  │  │  EVALUATOR  │  "Do these chunks answer     │            │
│  │  │  (LLM)      │   the query fully?"          │            │
│  │  │             │                              │            │
│  │  │  Output:    │                              │            │
│  │  │  sufficient: true/false                    │            │
│  │  │  missing: "needs example"                  │            │
│  │  │  refined_query: "sick leave                │            │
│  │  │    carry forward example calculation"      │            │
│  │  │  confidence: 0.6                           │            │
│  │  └──────┬──────┘                              │            │
│  │         │                                     │            │
│  └─────────┼─────────────────────────────────────┘            │
│            │                                                  │
│            ├── sufficient=true ──> Go to SYNTHESIZER           │
│            │                                                  │
│            └── sufficient=false ──┐                            │
│                                   │                           │
│  ┌─────────────── ITERATION 2 ───┼──────────────┐            │
│  │                                v              │            │
│  │  ┌───────────┐    refined_query:              │            │
│  │  │ RETRIEVER │    "sick leave carry forward   │            │
│  │  │           │     example calculation"        │            │
│  │  │ FAISS     │──> [C1] Leave policy details   │            │
│  │  │ (search   │    [C2] Accumulation rules     │            │
│  │  │  again)   │    ...                         │            │
│  │  └───────────┘                                │            │
│  │        │                                      │            │
│  │        v                                      │            │
│  │  ┌─────────────┐                              │            │
│  │  │  EVALUATOR  │  (evaluate again)            │            │
│  │  └─────────────┘                              │            │
│  │                                               │            │
│  └───────────────────────────────────────────────┘            │
│            │                                                  │
│            │  MAX 2 iterations, then synthesize                │
│            v                                                  │
│  ┌──────────────────────────────────────────────┐             │
│  │            SYNTHESIZER (LLM)                  │             │
│  │                                              │             │
│  │  INPUT: query + chunks + notes + limitations  │             │
│  │                                              │             │
│  │  Rules:                                       │             │
│  │  - Answer ONLY from provided chunks           │             │
│  │  - Cite every fact as [C1], [C2], etc.        │             │
│  │  - Follow formatting notes from Front Agent   │             │
│  │  - Acknowledge gaps if info is incomplete     │             │
│  │                                              │             │
│  │  OUTPUT:                                      │             │
│  │    answer: "The limit is 24 days [C1]..."     │             │
│  │    citations_used: ["[C1]", "[C3]"]           │             │
│  └──────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow (Single Query)

```
 USER INPUT                FRONT AGENT              AGENTIC RAG LOOP              OUTPUT
 ─────────                 ───────────              ────────────────              ──────

 "Explain sick    ┌──────────────────┐
  leave carry  -->│ Read Memory      │
  forward with    │ State            │
  an example."    │                  │
                  │ Resolve pronouns │    ┌─────────────────────────┐
                  │ Rewrite query    │--->│ ITERATION 1             │
                  │                  │    │                         │
                  │ RetrievalPlan:   │    │ Retrieve(query, k=8)    │
                  │  clean_query=    │    │     |                   │
                  │  "Explain sick   │    │     v                   │
                  │   leave carry    │    │ Evaluate sufficiency    │
                  │   forward with   │    │     |                   │
                  │   example"       │    │ sufficient=false        │
                  │  k=8             │    │ missing="needs example" │
                  │  notes=[]        │    │ refined_query="sick     │
                  └──────────────────┘    │  leave carry forward    │
                                          │  example calculation"   │
                                          └────────┬────────────────┘
                                                   |
                                          ┌────────v────────────────┐
                                          │ ITERATION 2             │
                                          │                         │
                                          │ Retrieve(refined, k=8)  │
                                          │     |                   │
                                          │     v                   │
                                          │ Evaluate sufficiency    │
                                          │     |                   │     ┌──────────────┐
                                          │ (max iterations reached)│     │              │
                                          │     |                   │---->│  SYNTHESIZE   │
                                          │     v                   │     │  with [Cx]    │
                                          │ Synthesize answer       │     │  citations    │
                                          └─────────────────────────┘     │              │
                                                                          │  "The sick   │
                                                   ┌──────────────┐       │  leave limit │
                                                   │ TRACER       │       │  is 24 days  │
                                                   │              │       │  [C1]..."    │
                                                   │ Save full    │       └──────┬───────┘
                                                   │ trace.json   │              │
                                                   │ with all     │              v
                                                   │ iterations   │      Display to User
                                                   └──────────────┘      + Update Memory
```

---

## Tracing (trace.json per query)

```
┌────────────────────────────────────────────────┐
│  trace.json                                     │
│                                                │
│  {                                              │
│    "run_id": "20260225T060728_2d81",            │
│    "user_message": "What is the sick...",       │
│    "memory_summary": "User asked about...",     │
│                                                │
│    "retrieval_plan": {                          │
│      "clean_query": "sick leave carry...",      │
│      "k": 8,                                   │
│      "notes": []                               │
│    },                                          │
│                                                │
│    "iteration_1": {                             │
│      "query": "sick leave carry forward...",    │
│      "retrieved": [                             │
│        {"chunk_id": "C1", "content": "...",     │
│         "source": "HR-POL-008...", "score": 0.4}│
│      ],                                        │
│      "evaluator": {                             │
│        "sufficient": true,                      │
│        "confidence": 0.95                       │
│      }                                         │
│    },                                          │
│                                                │
│    "iteration_2": null  (or filled if 2-pass), │
│                                                │
│    "final_answer": "The limit is 24 days [C1]" │
│    "citations_used": ["[C1]"]                   │
│  }                                              │
└────────────────────────────────────────────────┘
```

---

## N-Turn Memory Flow

```
  Turn 1                      Turn 2                      Turn 3
  ──────                      ──────                      ──────

  User: "What is the         User: "Explain with         User: "Now summarize
  sick leave limit?"          an example."                 that in 2 lines."
       │                           │                           │
       v                           v                           v
  Memory: (empty)            Memory:                     Memory:
                              history: [                  history: [
                                {user: Q1}                  {user: Q1}
                                {asst: A1}                  {asst: A1}
                              ]                             {user: Q2}
                              summary: "User asked          {asst: A2}
                               about sick leave           ]
                               carry forward limit"       summary: "User asked
                                                           about sick leave
                                   │                       carry forward, then
                                   v                       asked for example"
                              Front Agent resolves              │
                              context from memory               v
                                                          Front Agent:
                                                          "that" = sick leave
                                                           carry forward
                                                          notes: ["2 lines"]
```

---

## Project Structure Map

```
agentic_rag_project/
│
├── config/settings.py          # Env, paths, LLM/embedding factories
├── models/schemas.py           # ALL Pydantic data contracts
│
├── ingestion/                  # [Dev 1] Document Processing
│   ├── loader.py               #   .docx -> LangChain Documents
│   ├── chunker.py              #   Documents -> Semantic Chunks
│   ├── embedder.py             #   Chunks -> FAISS Vector Index
│   └── ingest_pipeline.py      #   Run: python -m ingestion.ingest_pipeline
│
├── memory/                     # [Dev 2] Conversation State
│   └── memory_model.py         #   History + Summary + Facts
│
├── agents/                     # [Dev 2] Query Planning
│   └── front_agent.py          #   Message + Memory -> RetrievalPlan
│
├── rag/                        # [Dev 3] Agentic Retrieval
│   ├── retriever.py            #   FAISS search -> RetrievedChunks
│   ├── evaluator.py            #   Chunks sufficient? -> Verdict
│   ├── synthesizer.py          #   Chunks -> Answer with [Cx] cites
│   └── agentic_loop.py         #   2-pass: retrieve->evaluate->refine
│
├── orchestrator/               # [Dev 4] Glue + Observability
│   ├── orchestrator.py         #   Wires all components
│   └── tracer.py               #   Saves per-query trace.json
│
├── app/                        # [Dev 4] User Interface
│   ├── run_chat.py             #   CLI: python -m app.run_chat
│   └── app.py                  #   Streamlit: streamlit run app/app.py
│
├── docs/                       # 10 HR policy .docx files
├── vector_store/               # Persisted FAISS index
└── traces/                     # Per-query trace.json output
```
