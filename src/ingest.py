"""
ingest.py — Load PDFs from data/, clean + chunk text, embed with nomic-embed-text,
and store everything in a local ChromaDB collection.

Run this once (or re-run after adding new PDFs). It won't re-embed docs that
are already in the store — it wipes and rebuilds from scratch each time, which
is the safest default for a corpus this size.
"""

import os
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── paths (relative to project root, not this file) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "nitrourkela_docs"

# ── embedding model ──────────────────────────────────────────────────────────
# nomic-embed-text produces 768-dim vectors; good balance of quality vs speed.
EMBED_MODEL = "nomic-embed-text"


def clean_text(text: str) -> str:
    """
    Light-touch cleanup for PDF-extracted text.

    PDFs often have:
      - repeated headers/footers on every page (e.g. "NIT Rourkela | Page 3")
      - hyphenated line-breaks ("informa-\ntion")
      - collapsed whitespace and stray newlines mid-sentence

    We don't go overboard — over-cleaning destroys meaning.
    """
    # Re-join words split across lines with a hyphen
    text = re.sub(r"-\n", "", text)

    # Collapse runs of whitespace / newlines that aren't paragraph breaks
    # (keep double-newlines so paragraphs stay separate)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip common boilerplate patterns (page numbers, "NIT Rourkela" headers)
    text = re.sub(r"(?i)(nit\s+rourkela\s*[\|–\-]?\s*page\s*\d+)", "", text)
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)

    return text.strip()


def load_pdfs(data_dir: Path) -> list:
    """
    Recursively find every PDF under data_dir and load it with PyPDFLoader.
    Returns a flat list of LangChain Document objects (one per PDF page).
    """
    pdf_paths = sorted(data_dir.rglob("*.pdf"))

    if not pdf_paths:
        print(f"[ingest] No PDFs found in {data_dir}. Drop some files in data/ and retry.")
        sys.exit(1)

    print(f"[ingest] Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"         • {p.relative_to(data_dir)}")

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()  # each page → one Document

        for page in pages:
            # Clean extracted text in place
            page.page_content = clean_text(page.page_content)

            # Normalise metadata so downstream code can rely on these keys
            page.metadata["source_file"] = pdf_path.name
            # PyPDFLoader sets metadata["page"] as a 0-based int; keep it
            page.metadata["page"] = page.metadata.get("page", 0)

        # Drop pages that are essentially empty after cleaning
        non_empty = [p for p in pages if len(p.page_content.strip()) > 50]
        all_docs.extend(non_empty)

    print(f"[ingest] Loaded {len(all_docs)} non-empty page(s) total.")
    return all_docs


def chunk_documents(docs: list) -> list:
    """
    Split pages into smaller chunks for retrieval.

    chunk_size=600 chars keeps chunks focused on a single topic.
    overlap=80 ensures sentences don't get arbitrarily cut at boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        # Prefer splitting on paragraph breaks, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Tag each chunk with its index within its source page — useful for debugging
    chunk_counter: dict[str, int] = {}
    for chunk in chunks:
        key = f"{chunk.metadata['source_file']}::{chunk.metadata['page']}"
        idx = chunk_counter.get(key, 0)
        chunk.metadata["chunk_index"] = idx
        chunk_counter[key] = idx + 1

    print(f"[ingest] Split into {len(chunks)} chunk(s).")
    return chunks


def build_vector_store(chunks: list) -> None:
    """
    Embed chunks with nomic-embed-text (via local Ollama) and persist to ChromaDB.

    We delete + recreate the collection on every run so the index stays in sync
    with whatever is currently in data/. For large corpora you'd want incremental
    upserts — overkill for this use case.
    """
    print(f"[ingest] Connecting to Ollama and loading '{EMBED_MODEL}' embeddings …")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Wipe any previous collection so we start fresh
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("[ingest] Cleared existing ChromaDB index.")

    print("[ingest] Embedding + indexing chunks (this may take a minute) …")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    count = vectorstore._collection.count()
    print(f"\n[ingest] ✓ Done. {count} chunk(s) stored in ChromaDB at {CHROMA_DIR}.")
    print( "[ingest]   Run `uvicorn src.api:app --reload` to start the API.")


def main():
    print("=" * 60)
    print(" NIT Rourkela RAG — Document Ingestion")
    print("=" * 60)

    docs   = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)


if __name__ == "__main__":
    main()
