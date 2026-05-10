from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel

from .chain import answer_question

logger = logging.getLogger(__name__)

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return answer_question(question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.exception("Ollama request failed.")
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not reachable. Ensure it is running.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering question.")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while generating the answer.",
        ) from exc
