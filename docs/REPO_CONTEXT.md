# Benjamin Repository Context

Last updated: 2026-02-17

## Purpose
- Internal strategy consulting assistant for three workflows:
- `V1` / `expert_network_brief` (Objective 1)
- `V2` / `interview_guide` (Objective 2)
- `V3` / `insights_qa` (Objective 3)
- Local-first RAG with strict provider controls for any cloud call.

## Current Runtime Implementation
- `backend.py` provides:
- `POST /api/chat` for direct and RAG flows
- `POST /api/chat/compare` for side-by-side Objective 3 compare (local + Opus)
- `GET /api/documents`
- `GET /api/documents/{doc_id}/file`
- `GET /api/stats`
- `GET /` (serves UI)
- OpenRouter requests enforce:
- `order: ["amazon-bedrock"]`
- `zdr: true`
- `allow_fallbacks: false`
- Objective routing:
- Obj 1 and Obj 2 use OpenRouter Sonnet (`anthropic/claude-sonnet-4.5`)
- Obj 3 uses local LM Studio by default (`LOCAL_RAG_MODEL`)
- Obj 3 compare mode also runs OpenRouter Opus (`OPUS_COMPARE_MODEL`, default `anthropic/claude-opus-4.6`)

## Current UI Behavior
- `static/index.html` has a 2x2 card layout:
- Top-left: request form
- Top-right: Library
- Bottom-left: Local response
- Bottom-right: Opus response (Objective 3 compare mode)
- Response cards show:
- token counts, model time, tok/sec
- retrieval timing
- RAG summary
- color-coded source confidence + snippets

## Ingestion and Data Tools
- `preprocess.py`: ingest/list/remove/stats for V1/V2/V3.
- The "Sync Data" / "Refresh" UI button directly integrates with `preprocess.py` to auto-ingest or remove data in the backend.
- `ingest_objective3.py`: Objective 3 wrapper (ingests `*.sanitized.txt` into `V3` only).
- `sanitize_with_lmstudio.py`: local de-identification pipeline.

## Key Constraints
- Use Python 3.12 (`chromadb` currently fails on Python 3.14 in this environment).
- Keep source corpora local; this repo intentionally keeps `Data/` gitignored.
- `.env` is the expected credential/config source:
- required: `OPENROUTER_API_KEY`
- optional: `LMSTUDIO_BASE_URL`, `LOCAL_RAG_MODEL`, `OPUS_COMPARE_MODEL`

## Canonical Docs
- `README.md`: primary user and operator documentation.
- `docs/IMPLEMENTATION_STATUS.md`: milestone completion status.
- `docs/NEXT_STEPS_CHECKLIST.md`: operational checklist and smoke tests.
- `IMPLEMENTATION_PLAN.md`: historical baseline plan.
- `RAG_IMPLEMENTATION_PLAN.md`: historical deep design plan.

## Quick Start
```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn backend:app --port 8000
```
