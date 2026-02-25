"""FAISS index build / save / load utilities."""

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings


def build_faiss_index(
    chunks: List[Document], embeddings: AzureOpenAIEmbeddings
) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    vs = FAISS.from_documents(chunks, embeddings)
    if vs is None:
        raise RuntimeError("FAISS.from_documents returned None.")
    print(f"[embedder] Built FAISS index with {len(chunks)} vectors")
    return vs


def save_index(vs: FAISS, path: Path) -> None:
    """Persist the FAISS index to disk."""
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    print(f"[embedder] Index saved to {path}")


def load_index(path: Path, embeddings: AzureOpenAIEmbeddings) -> FAISS:
    """Load a persisted FAISS index."""
    vs = FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )
    print(f"[embedder] Index loaded from {path}")
    return vs
