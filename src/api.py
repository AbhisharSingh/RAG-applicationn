"""
api.py — FastAPI application. Two endpoints, nothing fancy.

Start with:
    uvicorn src.api:app --reload --port 8000
"""

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chain import ask
from src.retriever import get_vectorstore


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up connections at startup so the first real request isn't slow.
    If ChromaDB or Ollama aren't ready we fail fast with a clear message.
    """
    print("[api] Warming up — loading vector store …")
    try:
        get_vectorstore()
        print("[api] ✓ Vector store loaded.")
    except FileNotFoundError as e:
        print(f"[api] ✗ {e}")
        # Don't crash the server — /health will report degraded and /ask will explain
    except Exception as e:
        print(f"[api] ✗ Unexpected error during startup: {e}")

    yield  # server is running

    print("[api] Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NIT Rourkela RAG API",
    description="Ask questions about NIT Rourkela institutional documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from a local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What are the admission criteria for B.Tech?")


class SourceRef(BaseModel):
    file: str
    page: int


class AnswerResponse(BaseModel):
    answer:  str
    sources: list[SourceRef]


class HealthResponse(BaseModel):
    status:       str
    vector_store: str
    ollama:       str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """
    Quick liveness + readiness check.
    Returns the status of both ChromaDB and Ollama so you know exactly
    what's broken if something is wrong.
    """
    # Check vector store
    try:
        vs    = get_vectorstore()
        count = vs._collection.count()
        vs_status = f"ok ({count} chunks indexed)"
    except FileNotFoundError:
        vs_status = "not ready — run `python -m src.ingest` first"
    except Exception as e:
        vs_status = f"error: {e}"

    # Check Ollama by hitting its local REST API
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
        ollama_status = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except httpx.ConnectError:
        ollama_status = "not running — start Ollama first"
    except Exception as e:
        ollama_status = f"error: {e}"

    overall = "ok" if "ok" in vs_status and ollama_status == "ok" else "degraded"

    return HealthResponse(status=overall, vector_store=vs_status, ollama=ollama_status)


@app.post("/ask", response_model=AnswerResponse, tags=["rag"])
async def ask_question(body: QuestionRequest):
    """
    Main RAG endpoint.

    Accepts a natural-language question, retrieves relevant document chunks,
    and returns an answer grounded in those chunks plus the source references.
    """
    try:
        result = ask(body.question)
    except FileNotFoundError as e:
        # Vector store not built yet
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        err = str(e)
        # Give the user an actionable message for the most common failure mode
        if "connection refused" in err.lower() or "connect error" in err.lower():
            raise HTTPException(
                status_code=503,
                detail="Cannot reach Ollama. Make sure it's running (`ollama serve`) and the required models are pulled.",
            )
        raise HTTPException(status_code=500, detail=f"Unexpected error: {err}")

    return AnswerResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
    )
