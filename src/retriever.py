from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT_DIR / "chroma_db"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@lru_cache
def _load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. Run ingest.py first."
        )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def retrieve(query: str, k: int = 4):
    if not query.strip():
        return []

    vectorstore = _load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        raise ValueError("Vector store is empty. Run ingest.py to index documents.")

    return docs
