"""

"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from src.retriever import retrieve

LLM_MODEL = "llama3.2"



PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant for NIT Rourkela. Your job is to answer
questions about the institute using ONLY the document excerpts provided below.

Rules you must follow:
- If the answer is clearly present in the context, answer it directly and cite the source file(s).
- If the answer is NOT in the context, say: "I couldn't find information about that in the available documents."
- Never make up facts, dates, names, or numbers that aren't in the context.
- Keep your answer focused and to the point.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer:""",
)


_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        # temperature=0 → deterministic answers, no hallucination-friendly randomness
        _llm = OllamaLLM(model=LLM_MODEL, temperature=0)
    return _llm


def format_context(chunks: list[dict]) -> str:
    """
    Turn retrieved chunks into a readable context block.
    Each chunk is labelled with its source so the LLM can (and does) cite it.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}] {chunk['source_file']} (page {chunk['page'] + 1})"
        parts.append(f"{label}\n{chunk['content']}")
    return "\n\n".join(parts)


def ask(question: str, k: int = 4) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.

    Args:
        question: The user's question.
        k:        How many chunks to retrieve (passed to retriever).

    Returns:
        {
            "answer":  str,
            "sources": [{"file": str, "page": int}, ...]   # deduplicated
        }
    """
    # Step 1 — semantic search
    chunks = retrieve(question, k=k)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
        }

   
    context = format_context(chunks)

   
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    llm    = get_llm()
    answer = llm.invoke(prompt)

    
    seen    = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source_file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": chunk["source_file"],
                "page": chunk["page"] + 1,  # return 1-based to humans
            })

    return {
        "answer":  answer.strip(),
        "sources": sources,
    }
