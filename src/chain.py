from __future__ import annotations

from functools import lru_cache
import os

from langchain_ollama import OllamaLLM

from .retriever import retrieve

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

PROMPT_TEMPLATE = """You are a QA assistant for NIT Rourkela documents.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't know based on the provided documents."
Never fabricate information.

Context:
{context}

Question: {question}
Answer:"""


@lru_cache
def _llm() -> OllamaLLM:
    return OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


def answer_question(question: str, k: int = 4):
    docs = retrieve(question, k=k)
    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    response = _llm().invoke(PROMPT_TEMPLATE.format(context=context, question=question))

    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"file": source, "page": int(page) if page is not None else -1})

    return {"answer": response.strip(), "sources": sources}
