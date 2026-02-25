"""Load .docx HR policy documents."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document


def load_documents(docs_dir: Path) -> List[Document]:
    """Load all .docx files from *docs_dir* and return LangChain Documents."""
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    files = sorted(docs_dir.glob("*.docx"))
    if not files:
        raise FileNotFoundError(f"No .docx files in: {docs_dir}")

    docs: List[Document] = []
    for fp in files:
        loaded = Docx2txtLoader(str(fp)).load()
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["source"] = fp.name
        docs.extend(loaded)

    print(f"[loader] Loaded {len(docs)} document(s) from {len(files)} file(s)")
    return docs
