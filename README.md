# RAG-applicationn

## What it is
A RAG-based Q&A API for NIT Rourkela institutional PDFs (placement brochures, academic calendars, rulebooks). It uses Ollama locally for both embeddings and the LLM, ChromaDB for vector storage, and FastAPI for serving.

## Setup
1. Install and start Ollama:
   - `ollama serve`
2. Pull the models:
   - `ollama pull llama3.2`
   - `ollama pull nomic-embed-text`
3. Create a virtual environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`

Optional environment overrides:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_LLM_MODEL` (default: `llama3.2`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `CHUNK_SIZE` (default: `600`)
- `CHUNK_OVERLAP` (default: `80`)
- `RESET_CHROMA` (default: `true`)

## Ingest documents
1. Place PDFs under `data/` (nested folders are supported).
2. Run ingestion:
   - `python -m src.ingest`

This creates a persisted Chroma index in `chroma_db/`.

## Run the API
Start the server:
- `uvicorn src.api:app --reload`

Health check:
- `GET http://localhost:8000/health`

## Query the API
Example request:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the eligibility criteria for placements?"}'
```
