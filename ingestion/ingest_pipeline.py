"""Orchestrates load → chunk → embed → persist.

Run standalone:  python -m ingestion.ingest_pipeline
"""

from config.settings import DOCS_DIR, VECTOR_STORE_DIR, make_embeddings
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import build_faiss_index, save_index


def run_ingestion() -> None:
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60)

    embeddings = make_embeddings()
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs, embeddings)
    vs = build_faiss_index(chunks, embeddings)
    save_index(vs, VECTOR_STORE_DIR)

    print("=" * 60)
    print(f"Done. {len(chunks)} chunks indexed -> {VECTOR_STORE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
