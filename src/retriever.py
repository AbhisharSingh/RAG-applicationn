"""
retriever.py — Thin wrapper around the persisted ChromaDB index.


"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
CHROMA_DIR      = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "nitrourkela_docs"
EMBED_MODEL     = "nomic-embed-text"


def _load_vectorstore() -> Chroma:
    """
    Open the persisted ChromaDB collection.
    Raises a clear error if ingestion hasn't been run yet.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run `python -m src.ingest` first to index your documents."
        )

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )



_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _load_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 4) -> list[dict]:
    """
    Semantic search over the indexed chunks.

    Args:
        query: The user's question (natural language).
        k:     Number of chunks to return. 4 is a sensible default —
               enough context without blowing the LLM's context window.

    Returns:
        List of dicts, each with:
          - "content"     : the chunk text
          - "source_file" : original PDF filename
          - "page"        : 0-based page number from the PDF
          - "chunk_index" : chunk position within that page
          - "score"       : cosine similarity score (higher = more relevant)
    """
    vs = get_vectorstore()

    # similarity_search_with_score returns (Document, score) tuples
    results = vs.similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        chunks.append({
            "content":     doc.page_content,
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page":        doc.metadata.get("page", 0),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score":       round(float(score), 4),
        })

    return chunks
