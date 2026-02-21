# Benjamin — Step-by-Step Implementation Plan

> Status note (2026-02-17): This is the original implementation plan and is kept for historical traceability.
> Current runtime behavior, API routes, and operator instructions are documented in:
> - `README.md`
> - `docs/IMPLEMENTATION_STATUS.md`
> - `docs/REPO_CONTEXT.md`

A milestone-driven implementation plan where each milestone has tests that must pass before proceeding to the next.

---

## What Exists Today

- `backend.py` — FastAPI server with direct + RAG chat, Objective routing, and compare endpoint ✅
- `static/index.html` — objective-aware UI with library + local/Opus response panels ✅
- `preprocess.py` + `ingest_objective3.py` — ingestion/management CLI paths ✅
- `rag/` modules + `system_prompts/` + tests are implemented ✅
- This plan’s remaining sections describe the original sequencing used to get here.

## What Needs to Be Built

RAG infrastructure, preprocessing CLI, system prompts, updated backend, updated frontend — all local.

---

## Milestone 1: Project Scaffolding & Dependencies

**Build:**
1. Update `requirements.txt` — add sentence-transformers, chromadb, tiktoken, pypdf, psutil, torch, pytest
2. Create directory structure: `rag/`, `system_prompts/`, `tests/`
3. Create `rag/__init__.py` (empty)
4. Create `.gitignore` entries for `chroma_db/`, `uploaded_docs/`, `.venv/`, `__pycache__/`

**Tests (gate):**
```bash
# T1.1: All deps install cleanly
pip install -r requirements.txt

# T1.2: Directory structure exists
python -c "from pathlib import Path; assert Path('rag/__init__.py').exists(); assert Path('system_prompts').is_dir(); assert Path('tests').is_dir(); print('PASS')"
```

---

## Milestone 2: Chunking Module (`rag/chunking.py`)

**Build:**
1. `chunk_text(text, chunk_size=512, overlap=50) -> list[str]` — token-based chunking using tiktoken
2. Skip chunks smaller than `MIN_CHUNK_SIZE=100` tokens
3. Return list of text strings

**Tests (gate):** `tests/test_chunking.py`
```
T2.1: Empty string → returns []
T2.2: Short text (< 100 tokens) → returns [] (below min)
T2.3: Text of exactly 512 tokens → returns [one chunk]
T2.4: Text of 1024 tokens with overlap=50 → returns 2+ chunks, verify overlap exists
T2.5: Very long text (5000 tokens) → verify chunk count is correct, all chunks ≤ 512 tokens
T2.6: Round-trip — concatenated chunks contain all original content (no data loss)
```
```bash
pytest tests/test_chunking.py -v
```

---

## Milestone 3: Embedding Module (`rag/embeddings.py`)

**Build:**
1. `get_embedding_model() -> SentenceTransformer` — singleton, lazy-loaded, MPS device with CPU fallback
2. `get_embedding(text) -> list[float]` — single text → 768-dim vector
3. `get_embeddings_batch(texts, batch_size=32) -> list[list[float]]` — batch embedding

**Tests (gate):** `tests/test_embeddings.py`
```
T3.1: get_embedding("hello") returns list of 768 floats
T3.2: get_embeddings_batch(["a", "b", "c"]) returns 3 vectors, each 768-dim
T3.3: Same text produces same embedding (deterministic)
T3.4: Different texts produce different embeddings
T3.5: Singleton — calling get_embedding_model() twice returns same object
```
```bash
pytest tests/test_embeddings.py -v
```

---

## Milestone 4: Vector Store Module (`rag/vector_store.py`)

**Build:**
1. `VectorStore(persist_path)` — wraps ChromaDB PersistentClient
2. `create_collection(name)` / `get_collection(name)` — one collection per vertical
3. `upsert_document(collection_name, doc_id, chunks, embeddings, metadata)`
4. `search(collection_name, query_embedding, top_k, min_score) -> list[dict]`
5. `delete_document(collection_name, doc_id)`
6. `get_stats(collection_name) -> dict`
7. `list_documents(collection_name) -> list[dict]`

**Tests (gate):** `tests/test_vector_store.py` (uses temp directory for Chroma)
```
T4.1: Create collection, verify it exists
T4.2: Upsert 3 chunks for doc_id "test_1", verify count == 3
T4.3: Search with a query vector, verify results returned with scores
T4.4: Delete doc_id "test_1", verify count == 0
T4.5: Upsert same doc_id twice (idempotent), verify count unchanged
T4.6: Two collections (V1, V2) are independent — data in V1 not visible in V2
T4.7: get_stats returns correct vector count and doc count
T4.8: Persistence — close and reopen store, data still present
```
```bash
pytest tests/test_vector_store.py -v
```

---

## Milestone 5: Document Manager (`rag/document_manager.py`)

**Build:**
1. `DocumentManager(storage_path)` — manages SQLite metadata + file storage
2. `save_document(file_bytes, filename, vertical) -> doc_id` — content-hash-based ID
3. `get_document(doc_id) -> dict`
4. `list_documents(vertical=None) -> list[dict]`
5. `delete_document(doc_id)` — removes file + metadata
6. `is_unchanged(file_bytes, filename) -> bool` — content-hash comparison for update detection
7. SQLite schema: `doc_id, filename, vertical, file_type, file_size, chunk_count, content_hash, ingested_at`

**Tests (gate):** `tests/test_document_manager.py` (uses temp directory)
```
T5.1: save_document returns a doc_id starting with "doc_"
T5.2: get_document returns correct metadata
T5.3: list_documents filters by vertical
T5.4: delete_document removes file and metadata
T5.5: is_unchanged returns True for same content, False for different content
T5.6: Re-saving same file returns same doc_id (idempotent)
T5.7: Original file is preserved on disk
```
```bash
pytest tests/test_document_manager.py -v
```

---

## Milestone 6: Retrieval Module (`rag/retrieval.py`)

**Build:**
1. `retrieve_context(query, vector_store, embedding_model, collection_name, top_k=5, min_score=0.5) -> list[dict]`
2. `assemble_context(chunks) -> str` — formats chunks as XML-tagged context for the LLM
3. Deduplication of overlapping chunks from same document

**Tests (gate):** `tests/test_retrieval.py`
```
T6.1: Ingest a known document, query with related text, verify relevant chunks returned
T6.2: min_score filtering — low-relevance results excluded
T6.3: top_k limiting — never returns more than top_k results
T6.4: assemble_context produces valid XML-tagged string with document name and chunk info
T6.5: Query against empty collection returns []
T6.6: Collection filtering — query V1 does not return V2 results
```
```bash
pytest tests/test_retrieval.py -v
```

---

## Milestone 7: Preprocessing CLI (`preprocess.py`)

**Build:**
1. CLI with argparse: `python preprocess.py --vertical V1 --files file1.docx file2.txt`
2. Also supports: `python preprocess.py --vertical V1 --dir ./briefs/`
3. For each file: read → chunk → embed → upsert to vector store → save metadata
4. `--remove doc_id` flag to remove a single document
5. `--list [vertical]` flag to show ingested documents
6. `--stats` flag to show storage stats
7. Skip unchanged files (content-hash check), print skip message
8. Progress output: filename, chunk count, time elapsed per file

**Tests (gate):** `tests/test_preprocess.py`
```
T7.1: Ingest a .txt file into V1, verify it appears in --list V1
T7.2: Ingest a .docx file into V2, verify chunk count > 0
T7.3: Re-ingest same file, verify "skipped (unchanged)" message
T7.4: Modify file content, re-ingest, verify chunks updated
T7.5: --remove doc_id, verify document gone from --list and vector store
T7.6: --dir ingests all supported files in directory
T7.7: --stats shows correct counts per vertical
T7.8: Unsupported file type is skipped with warning
```
```bash
pytest tests/test_preprocess.py -v
```

**Manual verification:**
```bash
# Create test files
echo "This is a sample expert network brief about the semiconductor industry." > /tmp/test_brief.txt
python preprocess.py --vertical V1 --files /tmp/test_brief.txt
python preprocess.py --list V1
python preprocess.py --stats
```

---

## Milestone 8: System Prompts

**Build:**
1. `system_prompts/expert_network_brief.md` — Objective 1 prompt (from PRD Section 6.1)
2. `system_prompts/interview_guide.md` — Objective 2 prompt (from PRD Section 6.2)
3. Helper: `load_system_prompt(objective) -> str` in backend or a small utility

**Tests (gate):**
```
T8.1: Both .md files exist and are non-empty
T8.2: load_system_prompt("expert_network_brief") returns string containing "screening"
T8.3: load_system_prompt("interview_guide") returns string containing "stakeholder"
T8.4: load_system_prompt("nonexistent") raises FileNotFoundError
```
```bash
pytest tests/test_system_prompts.py -v
```

---

## Milestone 9: Backend — RAG Chat Endpoint

**Build:**
1. Extend `backend.py` with RAG mode in `/api/chat`:
   - New form fields: `objective` (expert_network_brief | interview_guide), `mode` (direct | rag)
   - When `mode=rag`: retrieve from corresponding vertical → load system prompt → assemble context → call OpenRouter with system message + user message
   - Return response + source citations (doc_id, chunk_id, score)
2. Keep existing `mode=direct` behavior (backward compatible)
3. Add `/api/documents` GET endpoint (list ingested docs)
4. Add `/api/stats` GET endpoint
5. Startup event: initialize vector store + document manager (lazy-load embedding model)

**Tests (gate):** `tests/test_backend.py` (using FastAPI TestClient, mocked OpenRouter)
```
T9.1: POST /api/chat with mode=direct + file still works (backward compat)
T9.2: POST /api/chat with mode=rag + objective=expert_network_brief returns response with sources
T9.3: POST /api/chat with mode=rag but empty vertical returns graceful error
T9.4: GET /api/documents returns list
T9.5: GET /api/stats returns dict with expected keys
T9.6: System prompt is included in the OpenRouter payload as system message
T9.7: Provider config always includes zdr=true, order=["amazon-bedrock"], allow_fallbacks=false
T9.8: No network calls other than OpenRouter (security check on mocked requests)
```
```bash
pytest tests/test_backend.py -v
```

---

## Milestone 10: Frontend — Objective Selection & RAG Mode

**Build:**
1. Update `static/index.html`:
   - Objective selector (radio/dropdown): "Expert Network Brief" / "Interview Guide"
   - Mode toggle: Direct (file upload) vs. RAG (use ingested documents)
   - In RAG mode: show context input fields (project objectives, key questions, industry, stakeholder type)
   - In Direct mode: existing file upload behavior
   - Display source citations below response
   - Document library view (list ingested docs from `/api/documents`)
2. Keep the existing dark theme and design language

**Tests (gate):** Manual + automated
```
T10.1: Page loads without JS errors
T10.2: Objective selector switches between brief and interview guide
T10.3: Mode toggle shows/hides appropriate input fields
T10.4: RAG mode submission sends correct form fields (objective, mode, message)
T10.5: Direct mode submission still works as before
T10.6: Source citations display when RAG response includes them
T10.7: Document library populates from /api/documents
```
```bash
# Start server and verify manually
uvicorn backend:app --port 8000
# Then open http://localhost:8000 and test each scenario
```

---

## Milestone 11: End-to-End Integration Test

**Build:** `tests/test_e2e.py`

**Tests (gate):**
```
T11.1: Full pipeline — ingest test doc via preprocess.py → query via /api/chat mode=rag → verify response references ingested content
T11.2: Incremental update — add a second doc → query returns chunks from both docs
T11.3: Remove doc — delete via preprocess.py --remove → query no longer returns its chunks
T11.4: Cross-vertical isolation — doc in V1 not retrieved when querying V2
T11.5: Security audit — mock requests.post, verify only openrouter.ai is called, verify ZDR params present
T11.6: Restart persistence — stop server, restart, verify docs still in vector store
```
```bash
pytest tests/test_e2e.py -v
```

---

## Milestone 12: Cleanup & Documentation

**Build:**
1. Remove legacy files: `openrouter_sonnet_bedrock_zdr_test.py`, `plot_spx_index.py`
2. Clean up `requirements.txt` — remove yfinance, pandas, matplotlib
3. Update `RAG_IMPLEMENTATION_PLAN.md` to mark completed items
4. Final `README.md` review — ensure PRD matches implemented reality
5. Add `.gitignore` if not already present

**Tests (gate):**
```
T12.1: All pytest tests pass: pytest tests/ -v
T12.2: Server starts cleanly: uvicorn backend:app --port 8000
T12.3: No import errors: python -c "from rag.embeddings import get_embedding; from rag.vector_store import VectorStore; from rag.chunking import chunk_text; from rag.retrieval import retrieve_context"
```

---

## Execution Rules

1. **Complete each milestone fully before starting the next**
2. **All tests for a milestone must pass before proceeding**
3. **If a test fails, fix the implementation — do not skip or weaken the test**
4. **Run the full test suite (`pytest tests/ -v`) after every milestone to catch regressions**
