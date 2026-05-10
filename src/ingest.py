from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RESET_CHROMA = os.getenv("RESET_CHROMA", "true").lower() in {"1", "true", "yes"}

MIN_REPEAT_COUNT = 2
REPEAT_FRACTION = 0.5
MAX_REPEAT_LINE_LENGTH = 120


def _find_repeated_lines(page_lines: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in page_lines:
        counts.update(set(line.lower() for line in lines if line))
    threshold = max(MIN_REPEAT_COUNT, int(len(page_lines) * REPEAT_FRACTION))
    return {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) <= MAX_REPEAT_LINE_LENGTH
    }


def _clean_text(text: str, repeated_lines: set[str]) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        normalized = line.lower()
        if normalized in repeated_lines:
            continue
        if re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", normalized):
            continue
        cleaned.append(line)
    cleaned_text = re.sub(r"\s+", " ", " ".join(cleaned)).strip()
    return cleaned_text or re.sub(r"\s+", " ", text).strip()


def ingest() -> None:
    pdf_paths = sorted(DATA_DIR.rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found under {DATA_DIR}.")
        return

    if CHROMA_DIR.exists() and RESET_CHROMA:
        # Reset the index to start fresh and remove all existing documents (data loss).
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    total_pages = 0

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        total_pages += len(pages)

        page_lines = [
            [line.strip() for line in page.page_content.splitlines() if line.strip()]
            for page in pages
        ]
        repeated_lines = _find_repeated_lines(page_lines)

        for page in pages:
            page.page_content = _clean_text(page.page_content, repeated_lines)
            page.metadata["source"] = pdf_path.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

        chunks = splitter.split_documents(pages)
        chunk_index_by_page: defaultdict[tuple[str | None, int | None], int] = defaultdict(
            int
        )
        for chunk in chunks:
            key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
            chunk.metadata["chunk"] = chunk_index_by_page[key]
            chunk_index_by_page[key] += 1
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted from the PDFs.")
        return

    vectorstore = Chroma.from_documents(
        all_chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    print(
        "Ingestion complete: "
        f"{len(pdf_paths)} PDFs, {total_pages} pages, {len(all_chunks)} chunks "
        f"stored in {CHROMA_DIR}."
    )


if __name__ == "__main__":
    ingest()
