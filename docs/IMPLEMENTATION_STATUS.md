# Implementation Status vs Plan

Reference: `IMPLEMENTATION_PLAN.md`

## Summary
- Overall status: core milestones are now implemented in code.
- RAG modules, preprocessing CLI, prompts, backend RAG mode, frontend mode toggle, and test suite are in place.
- Full local test suite currently passes (`57 passed`).

## Milestone Tracking

### Milestone 1: Scaffolding & Dependencies
- Status: `COMPLETED`
- Added:
- `rag/`, `system_prompts/`, `tests/`
- RAG dependencies in `requirements.txt`
- `.gitignore` entries for local data and Python artifacts

### Milestone 2: Chunking Module
- Status: `COMPLETED`
- Implemented: `rag/chunking.py`, `tests/test_chunking.py`

### Milestone 3: Embedding Module
- Status: `COMPLETED`
- Implemented: `rag/embeddings.py`, `tests/test_embeddings.py`
- Notes: singleton loader with MPS/CPU fallback

### Milestone 4: Vector Store Module
- Status: `COMPLETED`
- Implemented: `rag/vector_store.py`, `tests/test_vector_store.py`
- Notes: collection normalization handles short names like `V1`/`V2`

### Milestone 5: Document Manager
- Status: `COMPLETED`
- Implemented: `rag/document_manager.py`, `tests/test_document_manager.py`

### Milestone 6: Retrieval Module
- Status: `COMPLETED`
- Implemented: `rag/retrieval.py`, `tests/test_retrieval.py`

### Milestone 7: Preprocessing CLI
- Status: `COMPLETED`
- Implemented: `preprocess.py`, `tests/test_preprocess.py`
- Includes: `--files`, `--dir`, `--remove`, `--list`, `--stats`

### Milestone 8: System Prompts
- Status: `COMPLETED`
- Implemented:
- `system_prompts/expert_network_brief.md`
- `system_prompts/interview_guide.md`
- `rag/system_prompts.py`
- `tests/test_system_prompts.py`

### Milestone 9: Backend RAG Endpoint
- Status: `COMPLETED`
- Implemented in `backend.py`:
- `POST /api/chat` supports `mode=direct|rag` and `objective`
- RAG retrieval + system prompt loading + citations in response
- `GET /api/documents`
- `GET /api/stats`
- Tests: `tests/test_backend.py`

### Milestone 10: Frontend Objective + RAG Mode
- Status: `COMPLETED` (core scope)
- Implemented in `static/index.html`:
- objective selector
- direct/rag mode toggle
- RAG context fields
- citations display
- document library panel via `/api/documents`

### Milestone 11: End-to-End Integration
- Status: `COMPLETED`
- Added: `tests/test_e2e.py`
- Covers ingest/query, incremental update, remove, isolation, provider config, persistence

### Milestone 12: Cleanup & Docs
- Status: `PARTIAL`
- Completed:
- removed `openrouter_sonnet_bedrock_zdr_test.py`
- removed legacy plotting/finance deps from `requirements.txt`
- Added and updated `docs/` context files
- Remaining:
- optional README refresh to fully reflect implemented runtime commands and architecture details

## Current Risks
- Python 3.14 is not compatible with current Chroma dependency; use Python 3.12 for now.
- Archived legacy virtualenv folders may still exist (`*_archived`) and can be deleted manually if desired.

## Recommended Immediate Focus
- Use the canonical `.venv` (Python 3.12) and run from that environment.
- Optionally update `README.md` with final implementation status and startup/test commands.
