# Next Steps Checklist (Post-Implementation)

## Phase A: Environment Standardization
1. Use Python 3.12 for this repo (`chromadb` currently fails on Python 3.14).
2. Recreate `.venv` only if needed:
```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```
3. Run full tests:
```bash
./.venv/bin/pytest tests/ -v
```

## Phase B: Runtime Smoke Checks
1. Start backend:
```bash
./.venv/bin/uvicorn backend:app --port 8000
```
2. Ingest sample content:
```bash
./.venv/bin/python preprocess.py --vertical V1 --files /tmp/test_brief.txt
./.venv/bin/python preprocess.py --list V1
./.venv/bin/python preprocess.py --stats
```

## Phase C: Quality / Product Follow-Ups
1. Tune retrieval quality (`top_k`, `min_score`, prompt format).
2. Add optional frontend actions for document delete/vertical filtering.
3. Refresh `README.md` to reflect implemented architecture and commands.
4. Add CI test workflow using Python 3.12.

## Definition of Done for This Repo
- All tests pass (`tests/`).
- Backend starts and serves UI.
- Direct mode works.
- RAG mode works with citations and vertical isolation.
- Only OpenRouter outbound call path is used by chat flow.
