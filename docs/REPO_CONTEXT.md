# Benjamin Repository Context

## Purpose
- Internal strategy consulting assistant that uses Claude Sonnet 4.5 through OpenRouter routed to Amazon Bedrock with ZDR.
- Current runtime pattern is direct chat with optional uploaded file context.
- Target state is local RAG with verticalized collections for:
- `V1`: expert network brief drafting
- `V2`: interview guide drafting

## Current Implementation (as of 2026-02-12)
- `backend.py`: FastAPI backend with:
- `POST /api/chat` supporting `mode=direct|rag`
- `GET /api/documents`
- `GET /api/stats`
- `GET /` static index page
- OpenRouter provider settings enforced on every request:
- `order: ["amazon-bedrock"]`
- `zdr: true`
- `allow_fallbacks: false`
- `static/index.html`: UI with objective selector, direct/rag mode toggle, RAG context fields, source citations, and document library panel.
- `preprocess.py`: CLI ingestion and management:
- `--files`, `--dir`, `--remove`, `--list`, `--stats`
- `rag/` modules implemented:
- `chunking.py`, `embeddings.py`, `vector_store.py`, `document_manager.py`, `retrieval.py`, `system_prompts.py`
- `system_prompts/` implemented:
- `expert_network_brief.md`
- `interview_guide.md`
- `tests/` implemented with unit, integration-style, and e2e coverage.

## Key Constraints
- Local-first processing is required by product design and security model.
- Only intended outbound call is OpenRouter chat completion.
- Credentials loaded via 1Password helper, env var, or local fallback env file path.

## Planned Target Architecture
- Core target architecture is now implemented.
- Remaining gap is mainly documentation polish and environment standardization.

## Existing Planning Docs
- `README.md`: PRD-style product and architecture requirements.
- `IMPLEMENTATION_PLAN.md`: milestone-gated delivery plan with tests.
- `RAG_IMPLEMENTATION_PLAN.md`: detailed technical design and rollout notes.

## Quick Start (Current Code)
```bash
pip install -r requirements.txt
uvicorn backend:app --port 8000
```

## Recommended Ground Rules for Upcoming Build
- Use Python 3.12 for runtime and tests.
- Keep direct mode backward compatible while iterating on RAG quality.
- Use tests as gates for retrieval quality and regression control.
