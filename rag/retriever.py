"""FAISS search wrapper → List[RetrievedChunk]."""

from __future__ import annotations

from typing import List

from langchain_community.vectorstores import FAISS

from models.schemas import RetrievedChunk


class Retriever:
    """Thin wrapper around a FAISS index that returns typed chunks with IDs."""

    def __init__(self, vector_store: FAISS) -> None:
        self._vs = vector_store

    def retrieve(self, query: str, k: int = 8) -> List[RetrievedChunk]:
        results = self._vs.similarity_search_with_score(query, k=k)
        chunks: List[RetrievedChunk] = []
        for idx, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"C{idx}",
                    content=doc.page_content,
                    source=meta.get("source", ""),
                    page=meta.get("page", 0),
                    score=float(score),
                )
            )
        return chunks
