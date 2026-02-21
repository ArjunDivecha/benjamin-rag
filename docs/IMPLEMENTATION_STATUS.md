# Implementation Status vs Plan

Reference baseline: `IMPLEMENTATION_PLAN.md`
Last updated: 2026-02-17

## Summary
- Core implementation is complete for objectives 1, 2, and 3.
- Objective 3 now supports side-by-side local vs Opus comparison in the UI and API.
- Documentation and operator workflows are aligned to current runtime behavior.

## Milestone Tracking

### Milestone 1: Scaffolding & Dependencies
- Status: `COMPLETED`
- `rag/`, `system_prompts/`, `tests/` present and active.
- Dependency set is focused on RAG/runtime needs.

### Milestone 2: Chunking Module
- Status: `COMPLETED`
- Implemented: `rag/chunking.py` with test coverage.

### Milestone 3: Embedding Module
- Status: `COMPLETED`
- Implemented: `rag/embeddings.py` with singleton loader and MPS/CPU fallback.

### Milestone 4: Vector Store Module
- Status: `COMPLETED`
- Implemented: `rag/vector_store.py` (Chroma persistent client, collection isolation).

### Milestone 5: Document Manager
- Status: `COMPLETED`
- Implemented: `rag/document_manager.py` (metadata + stored source file paths).

### Milestone 6: Retrieval Module
- Status: `COMPLETED`
- Implemented: `rag/retrieval.py` and context assembly helpers.

### Milestone 7: Preprocessing CLI
- Status: `COMPLETED`
- Implemented: `preprocess.py` with `--files`, `--dir`, `--remove`, `--list`, `--stats`.
- Objective 3 wrapper added: `ingest_objective3.py`.

### Milestone 8: System Prompts
- Status: `COMPLETED`
- Implemented:
- `system_prompts/expert_network_brief.md`
- `system_prompts/interview_guide.md`
- `system_prompts/insights_qa.md`
- `rag/system_prompts.py`

### Milestone 9: Backend RAG Endpoint
- Status: `COMPLETED`
- `POST /api/chat` supports direct + RAG mode and objective routing.
- Provider lock and ZDR controls are enforced on OpenRouter requests.
- RAG responses include usage, model/provider, metrics, and sources.

### Milestone 10: Frontend Objective + RAG Mode
- Status: `COMPLETED`
- Objective-specific UI behavior is implemented, including forced RAG for Objective 3.
- Current layout:
- top row: form + Library
- bottom row: local response + Opus response
- UI shows metrics, RAG summary, and confidence-colored source evidence.

### Milestone 11: End-to-End Integration
- Status: `COMPLETED`
- Integration coverage exists in `tests/test_e2e.py`.
- Backend compare coverage added for `POST /api/chat/compare`.

### Milestone 12: Cleanup & Docs
- Status: `COMPLETED`
- README and docs reflect:
- Python 3.12 requirement
- Objective 3 ingestion workflow
- local vs Opus compare flow
- API and UI behavior

## Current Runtime Notes
- Test suite size: `63 tests collected`.
- Objective 3 default local model is configurable via `LOCAL_RAG_MODEL`.
- Objective 3 compare cloud model is configurable via `OPUS_COMPARE_MODEL`.

## Known Constraints
- Use Python 3.12 for now due current Chroma compatibility.
- Local model latency can dominate Objective 3 response time on large contexts.
