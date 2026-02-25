"""Semantic chunking of documents."""

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings


def chunk_documents(
    docs: List[Document], embeddings: AzureOpenAIEmbeddings
) -> List[Document]:
    """Split *docs* into semantic chunks, preserving metadata."""
    splitter = SemanticChunker(embeddings=embeddings)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("SemanticChunker produced zero chunks.")
    print(f"[chunker] Produced {len(chunks)} chunk(s) from {len(docs)} document(s)")
    return chunks
