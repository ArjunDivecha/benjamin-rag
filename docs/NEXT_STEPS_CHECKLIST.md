# Operational Checklist

Last updated: 2026-02-17

## Phase A: Environment Baseline
1. Use Python 3.12 for this repo (`chromadb` currently fails on Python 3.14 in this environment).
2. Recreate `.venv` only if needed:
```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```
3. Ensure `.env` includes:
- `OPENROUTER_API_KEY=...` (required)
- `OLLAMA_BASE_URL=...` (optional, default `http://localhost:11434`)
- `LOCAL_RAG_MODEL=...` (optional)
- `OPUS_COMPARE_MODEL=...` (optional)

## Phase B: Data and Ingestion Checks
1. Ingest starter corpus if needed:
```bash
./.venv/bin/python preprocess.py --vertical V1 --dir synthetic_data/V1
./.venv/bin/python preprocess.py --vertical V2 --dir synthetic_data/V2
./.venv/bin/python ingest_objective3.py
```
2. Verify index state:
```bash
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --stats
```

## Phase C: Runtime Smoke Checks
1. Start backend:
```bash
./.venv/bin/uvicorn backend:app --port 8000
```
2. Validate Objective 1 or 2 basic call:
- Use UI or `POST /api/chat` with `mode=rag` and objective `expert_network_brief` or `interview_guide`.
3. Validate Objective 3 local call:
- Use `objective=insights_qa` and confirm local response + stats render.
4. Validate Objective 3 compare call:
- Use Insights Parlour in RAG mode and confirm both bottom panels populate:
- left: local response
- right: Opus response

## Phase D: Test Gates
1. Fast test pass:
```bash
./.venv/bin/pytest tests/test_backend.py -q
./.venv/bin/pytest tests/test_e2e.py -q
```
2. Full suite:
```bash
./.venv/bin/pytest tests/ -v
```

## Definition of Done
- Backend serves UI.
- Objective 1/2 RAG path works.
- Objective 3 local path works.
- Objective 3 compare path works.
- Metrics and sources render in UI.
- Tests pass in Python 3.12 environment.
