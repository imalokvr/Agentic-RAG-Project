"""Centralised configuration: env loading, paths, constants, LLM/embedding factories."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
TRACES_DIR = PROJECT_ROOT / "traces"

# ensure output dirs exist
VECTOR_STORE_DIR.mkdir(exist_ok=True)
TRACES_DIR.mkdir(exist_ok=True)

# ── env ────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

# ── RAG defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K = 8
MAX_RAG_ITERATIONS = 2

# ── Memory defaults ────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10


# ── factories ──────────────────────────────────────────────────────────
def make_llm(temperature: float = 0.2, max_tokens: int = 1024) -> AzureChatOpenAI:
    """Return a ready-to-use Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_embeddings() -> AzureOpenAIEmbeddings:
    """Return a ready-to-use Azure embeddings instance."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
