# Benjamin PRD — Strategy Consulting AI Assistant

Last updated: 2026-02-17

Benjamin is a confidential, RAG-powered AI assistant purpose-built for a boutique strategy consulting firm, using multi-vertical knowledge bases and custom system prompts to accelerate primary research workflows.

---

## Quick Start (Local)

1. Use Python 3.12.
2. Install dependencies:
   ```bash
   python3.12 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt
   ```
3. Create `.env` in the repo root:
   ```bash
   BEDROCK_API_KEY=your_bedrock_api_key_here
   # optional for Objective 3 local inference:
   OLLAMA_BASE_URL=http://localhost:11434
   LOCAL_RAG_MODEL=qwen2.5:32b
   # optional for Objective 3 side-by-side compare:
   OPUS_COMPARE_MODEL=us.anthropic.claude-opus-4-6-v1
   ```
4. Optional (Objective 3 local model): start Ollama and pull a model:
   ```bash
   ollama serve
   ollama pull qwen2.5:32b
   ```
5. Optional: ingest synthetic starter corpus:
   ```bash
   ./.venv/bin/python preprocess.py --vertical V1 --dir synthetic_data/V1
   ./.venv/bin/python preprocess.py --vertical V2 --dir synthetic_data/V2
   # Objective 3 only (ingests *.sanitized.txt from sanitized_data into V3):
   ./.venv/bin/python ingest_objective3.py
   ```
6. Start app:
   ```bash
   ./.venv/bin/uvicorn backend:app --port 8000
   ```
   or use the one-command launcher:
   ```bash
   ./run_benjamin.sh
   ```
7. In the UI:
- For Objective 3, use `With archived context` to run local RAG answers.
- Objective 3 compare renders **local** and **Opus** answers in adjacent bottom panels.

---

## TL;DR: Add/Update RAG Data (Copy/Paste)

```bash
# 1) Go to project
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Benjamin"

# 2) First-time setup only (skip if already done)
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 3) Create data folders
mkdir -p real_data/V1 real_data/V2 real_data/V3

# 4) Put your real files into:
#    real_data/V1  (expert network brief content)
#    real_data/V2  (interview guide content)
#    real_data/V3  (interview notes/transcripts for conversational Q&A)

# 5) Ingest / update RAG
./.venv/bin/python preprocess.py --vertical V1 --dir real_data/V1
./.venv/bin/python preprocess.py --vertical V2 --dir real_data/V2
# Objective 3 only: pull sanitized files from sanitized_data into V3
./.venv/bin/python ingest_objective3.py

# 6) Verify
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --stats
```

If you edit or add files later, run Step 5 again.

---

## One-Command Launcher

Use:

```bash
./run_benjamin.sh
```

What it does:
- ensures Python 3.12 is available
- creates `.venv` if missing
- installs dependencies when needed
- starts `uvicorn` on port `8000` (reload enabled)

Useful overrides:

```bash
PORT=8010 ./run_benjamin.sh
PYTHON_BIN=python3.12 ./run_benjamin.sh
./run_benjamin.sh --host 0.0.0.0 --port 9000 --reload
```

---

## Add or Update Real Project Data (Beginner Guide)

Use this exact workflow when your friend wants to add real files or update the RAG later.

### Step 0: Open terminal in the project folder

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Benjamin"
```

### Step 1: One-time setup (only first time on a new machine)

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Create `.env` in this folder with:

```bash
BEDROCK_API_KEY=your_bedrock_api_key_here
# Optional local-model settings for Objective 3
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_RAG_MODEL=qwen2.5:32b
# Optional Objective 3 compare model via AWS Bedrock
OPUS_COMPARE_MODEL=us.anthropic.claude-opus-4-6-v1
```

### Step 2: Create data folders

```bash
mkdir -p real_data/V1 real_data/V2 real_data/V3
```

Use:
- `real_data/V1` for Objective 1 content (expert network briefs, screening examples, past briefs)
- `real_data/V2` for Objective 2 content (interview guides, stakeholder question banks, methodology)
- `real_data/V3` for Objective 3 content (interview notes, transcripts, synthesis support documents)

### Step 3: Add files into the correct folder

Supported file types:
- `.txt`
- `.md`
- `.docx`
- `.pdf`
- `.csv`
- `.json`
- `.xml`
- `.yaml`
- `.yml`

Format expectations:
- Keep files text-heavy (the pipeline reads text, not images).
- For PDFs, prefer text-based PDFs (scanned/image-only PDFs often extract little or no text).
- Very short files may ingest as `0 chunks` and not help retrieval (chunks under 100 tokens are discarded).
- `--dir` is non-recursive (it only ingests files directly inside that folder).

### Step 4: Ingest into the RAG (this is the update command)

Run these commands every time you want the RAG to pick up new/changed files:

```bash
./.venv/bin/python preprocess.py --vertical V1 --dir real_data/V1
./.venv/bin/python preprocess.py --vertical V2 --dir real_data/V2
# Objective 3 only (recommended): ingest sanitized corpus from sanitized_data into V3
./.venv/bin/python ingest_objective3.py
```

You can also ingest specific files (useful if files are outside `real_data/`):

```bash
./.venv/bin/python preprocess.py --vertical V1 --files "/full/path/file1.docx" "/full/path/file2.pdf"
```

Objective 3 wrapper options:

```bash
# Preview which files would be ingested (no changes)
./.venv/bin/python ingest_objective3.py --dry-run

# If sanitized files are in a different folder:
./.venv/bin/python ingest_objective3.py --dir "/full/path/to/sanitized_data"
```

### Step 5: Confirm what is in the RAG

```bash
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --list V1
./.venv/bin/python preprocess.py --list V2
./.venv/bin/python preprocess.py --list V3
./.venv/bin/python preprocess.py --stats
```

What to expect in output:
- `ingested (...)` means the file was processed and vectors were updated.
- `skipped (unchanged)` means that exact file content was already ingested in that vertical.
- `skipped (unsupported file type)` means extension is not supported.
- `skipped (read error: ...)` means parsing failed for that file.

### Step 6: How to update data later

Common scenarios:

1. Replace an existing document with a newer version:
   - Keep the same filename in the same vertical folder.
   - Overwrite the file with new content.
   - Re-run Step 4 command for that vertical.
   - Result: old chunks for that filename are replaced automatically.

2. Add a brand-new document:
   - Drop it into `real_data/V1`, `real_data/V2`, or `real_data/V3`.
   - Re-run Step 4 command for that vertical.

3. Remove a document from the RAG:
   - Find its `doc_id`:
     ```bash
     ./.venv/bin/python preprocess.py --list
     ```
   - Remove by `doc_id`:
     ```bash
     ./.venv/bin/python preprocess.py --remove <doc_id>
     ```

4. Move a document across verticals (e.g., V1↔V2, V2↔V3, V1↔V3):
   - Remove old entry by `doc_id`.
   - Put file in the new vertical folder.
   - Re-run Step 4 for the new vertical.

---

## Local Data Sanitization (LM Studio, Fully Local)

Use this when you need de-identified copies of sensitive source files before sharing or downstream processing.

### What it does

- Reads supported files from an input directory: `.txt`, `.md`, `.docx`, `.pdf`, `.csv`, `.json`, `.xml`, `.yaml`, `.yml`
- Uses a local LM Studio model to extract person/org names per chunk
- Applies deterministic local replacement rules for identifiers:
  - Person names → `[PERSON]`
  - Organization names → `[ORG]`
  - Emails → `[EMAIL]`
  - Phones → `[PHONE]`
  - URLs → `[URL]`
  - IDs / long identifiers / SSN / IP → `[ID]`
- Writes sanitized outputs to a separate output directory
- Writes run metadata to `sanitization_report.json`

### Privacy defaults

- By default, output filenames are anonymized: `file_0001.sanitized.txt`, etc.
- By default, source paths are hashed in the report (not written in plaintext).
- To preserve source names/paths, pass `--preserve-names`.

### Prerequisites

1. LM Studio local server is running on `http://127.0.0.1:1234`
2. A model is loaded in LM Studio and reachable via `/v1/models`
3. Recommended model setup used in this repo:

```bash
$HOME/.lmstudio/bin/lms server start
$HOME/.lmstudio/bin/lms load qwen/qwen3-32b --identifier sanitizer_qwen --yes
```

### Recommended command (stable profile)

```bash
./.venv/bin/python sanitize_with_lmstudio.py \
  --input-dir Data \
  --output-dir sanitized_data \
  --model sanitizer_qwen \
  --use-nothink \
  --passes 1 \
  --max-chars 12000 \
  --overlap 500 \
  --chunk-workers 1
```

Notes:
- `--use-nothink` prefixes user prompts with `/nothink` for Qwen-style models.
- `--passes 1` is usually the best speed/quality tradeoff for large corpora.

### Resume and re-run behavior

- Resume incomplete runs:
  - Re-run the same command without `--overwrite`
  - Already-created sanitized files are skipped automatically
- Force full re-sanitization:
  - Add `--overwrite`

### Output files

- Sanitized documents: `sanitized_data/file_XXXX.sanitized.txt`
- Run report: `sanitized_data/sanitization_report.json`

### Build Objective 3 RAG from sanitized output

After sanitization finishes, run:

```bash
./.venv/bin/python ingest_objective3.py
./.venv/bin/python preprocess.py --list V3
./.venv/bin/python preprocess.py --stats
```

Important:
- `ingest_objective3.py` ingests only `*.sanitized.txt` files.
- It always writes to `V3` (Objective 3 collection only).
- `sanitization_report.json` and `sanitize_run.log` are not ingested by this wrapper.

Key report fields:
- `files_total`
- `files_succeeded`
- `files_skipped`
- `files_failed`
- Per-file `status`, `chunks`, `entities_people`, `entities_orgs`, timing, and errors

### Troubleshooting

1. Error: `400 ... No models loaded`
   - Cause: LM Studio server is up but no model is loaded
   - Fix:
   ```bash
   $HOME/.lmstudio/bin/lms load qwen/qwen3-32b --identifier sanitizer_qwen --yes
   ```

2. Warning: `Ignoring wrong pointing object ...`
   - Cause: recoverable PDF structure warnings from `pypdf`
   - Impact: usually non-fatal; run can still complete

3. Scanned/image PDFs sanitize poorly
   - Current pipeline is text-extraction based (no OCR fallback yet)
   - For best results, use text-based PDFs

4. Slow throughput on very large files
   - Reduce passes (`--passes 1`)
   - Increase chunk size (`--max-chars`)
   - Keep `--chunk-workers 1` for stability on this setup

---

## 1. Product Vision

Benjamin is an internal AI tool for a boutique strategy consulting firm that competes with McKinsey, BCG, and Bain. It accelerates repetitive components of client engagements across three workflows:

- **Objective 1**: Expert network brief drafting
- **Objective 2**: Interview guide drafting
- **Objective 3**: Conversational synthesis of interview notes and related research

**Client and company confidentiality is non-negotiable.** The current implementation uses a hybrid generation route:

- Objectives 1 and 2 use Claude Sonnet 3.5 via AWS Bedrock.
- Objective 3 defaults to a **local model via Ollama** over the same local RAG stack.
- Objective 3 also supports side-by-side comparison with AWS Bedrock Opus (same retrieved context).

---

## 2. Users & Context

- **Users**: Consultants conducting commercial due diligence and growth strategy engagements
- **Clients**: Investment firms, private equity firms, hedge funds, corporates
- **Industries**: Primarily technology and manufacturing, but cross-industry
- **Project types**: Commercial due diligence, growth strategy (market / customer / competitive analyses)
- **Research methods**: Primary research (interviews, surveys) and secondary research (databases, reports, filings)

---

## 3. Objectives

### Objective 1: Expert Network Brief Drafting

Generate project briefs for expert networks to identify and screen relevant stakeholders for interviews.

**Requirements:**
- Accept project objectives and key questions as input
- Output a structured brief suitable for expert network submission
- Include ~5 screening questions (+/- 1) that:
  - Cover a representative set of interview topics
  - Assess interviewees' level of knowledge
  - Extract useful information even if the person is not interviewed
- Tailor output to the specific industry, client type, and engagement scope

### Objective 2: Interview Guide Drafting

Generate tailored interview question lists for each stakeholder type (customers, competitors, internal employees, etc.).

**Requirements:**
- Accept project objectives, key questions, and stakeholder type as input
- Output a structured interview guide with questions optimized for:
  - Maximizing usefulness of responses
  - Maximizing interviewee willingness to speak
  - Phrasing sensitive questions to extract needed information without revealing intent
- Adapt tone and framing per stakeholder type

### Objective 3: Conversational Research Synthesis (Sensitive Notes)

Answer analyst-style questions over interview notes/transcripts and related research with explicit evidence.

**Representative question types:**
- "How many interviewees agreed vs disagreed with X?"
- "Find quotes supporting statement Y."
- "Who talked about topic Z?"
- "What is the consensus view on growth rate? Cite relevant references."

**Requirements:**
- Reuse the same RAG foundation as Objectives 1 and 2
- Route generation to a local LLM for sensitive content
- Return quote-level and source-level references
- Avoid unsupported claims when evidence is weak or conflicting

---

## 4. System Architecture

### 4.1 Preprocessing Pipeline (Offline / Batch)

A local preprocessing step ingests source documents and builds multiple RAG indices across verticals:

- **Input**: Past briefs, interview guides, methodology docs, interview notes, and related research artifacts
- **Processing**: Chunk -> embed (Sentence Transformers, MPS-accelerated) -> store in Chroma
- **Output**: Three named collections:
  - **V1** -> Objective 1 (briefs, screening examples, prior deliverables)
  - **V2** -> Objective 2 (interview guide patterns, stakeholder question structures)
  - **V3** -> Objective 3 (interview notes, transcripts, synthesis evidence)

The pipeline supports incremental updates. Re-ingesting an updated file replaces stale chunks for that filename+vertical.

### 4.2 Runtime Routing (Current)

| Flow | Collection | Provider | Model | Cloud egress |
|---|---|---|---|---|
| Obj 1 (`expert_network_brief`) | V1 | AWS Bedrock | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Yes |
| Obj 2 (`interview_guide`) | V2 | AWS Bedrock | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Yes |
| Obj 3 default (`POST /api/chat`) | V3 | Ollama (local) | `LOCAL_RAG_MODEL` (default `qwen2.5:32b`) | No |
| Obj 3 compare (`POST /api/chat/compare`) | V3 | Ollama + Bedrock | local `LOCAL_RAG_MODEL` + `OPUS_COMPARE_MODEL` (default `us.anthropic.claude-opus-4-6-v1`) | Yes (Opus leg) |

### 4.3 Web Application (Runtime)

A FastAPI + HTML frontend where consultants interact with three sections (Brief Salon, Interview Atelier, Insights Parlour):

1. Select objective
2. Choose mode
   - `direct`: optional file context + prompt
   - `rag`: retrieve from V1/V2/V3 + objective-specific prompt
3. Retrieve relevant chunks from the mapped vertical
4. Load objective-specific prompt from `system_prompts/*.md`
5. Route generation by objective:
   - Obj 1/2: AWS Bedrock
   - Obj 3 default: Ollama local
   - Obj 3 compare: Ollama local + Opus in parallel
6. Display answer with sources, token usage, and runtime stats

Current UX behavior:
- Objective 3 forces `rag` mode in the UI
- Objective-specific labels/placeholders/hints are applied dynamically
- Top row: form (left), **Library** (right)
- Bottom row: local response (left), Opus response (right, compare mode)
- Source cards are color-coded by confidence and include snippets

### 4.4 API Surface (Current)

- `POST /api/chat`
  - `mode=direct`: optional file context + prompt, AWS Bedrock generation
  - `mode=rag`: requires `objective` in `{expert_network_brief, interview_guide, insights_qa}`
  - RAG response includes `content`, `usage`, `provider`, `model`, `metrics`, `rag`, `sources`
- `POST /api/chat/compare`
  - Objective 3 compare only (`objective=insights_qa`)
  - Shared retrieval from V3, then runs local and Opus responses in parallel
  - Returns `local`, `opus`, shared `rag`, shared `sources`, and request-level metrics
- `GET /api/documents`
  - Lists ingested documents and metadata
- `GET /api/documents/{doc_id}/file`
  - Serves stored original files for Library links
- `GET /api/stats`
  - Returns per-vertical doc/chunk/vector stats

### 4.5 Security Model

```
LOCAL (consultant machine)                 CLOUD (Obj 1/2 + optional Obj 3 compare)
┌─────────────────────────────┐          ┌──────────────────────────┐
│ All source docs + uploads   │          │ AWS Bedrock API          │
│ All vectors + metadata      │   ->     │                          │
│ All preprocessing           │ prompt   │                          │
│ Web UI + FastAPI            │ only     └──────────────────────────┘
│ Ollama local model (Obj 3)  │          
└─────────────────────────────┘
```

- Objective 3 can run fully local (retrieval + generation)
- Objectives 1/2 and optional Objective 3 compare send only assembled prompts to AWS Bedrock
- Credential handling: API keys from local env files / environment variables only
- Repository hygiene: Raw source data folder `/Data/` is excluded from gitignored

---

## 5. Functional Requirements


| Requirement | Description |
|---|---|
| Multi-file ingestion | Accept a directory or explicit file list |
| Vertical assignment | Each file is tagged to `V1`, `V2`, or `V3` |
| Chunking | 512-token chunks with 50-token overlap; chunks under 100 tokens are dropped |
| Local embeddings | Sentence Transformers (`all-mpnet-base-v2`) |
| Persistent storage | Chroma (vectors) + SQLite (document metadata) |
| Idempotent update behavior | Same filename in same vertical is replaced on re-ingest |
| Update detection | Unchanged files are skipped by content hash |
| Management commands | List docs, remove by `doc_id`, and print stats |

### 5.2 Web Application

| Requirement | Description |
|---|---|
| Objective selection | User picks Objective 1, 2, or 3 |
| Mode selection | Direct upload mode and RAG mode (Obj 3 forces RAG) |
| RAG retrieval | Query mapped vertical (V1/V2/V3) with `top_k` and `min_score` |
| Objective prompts | Loads objective-specific system prompts from local files |
| Hybrid generation | Obj 1/2 via AWS Bedrock; Obj 3 via Ollama local; Obj 3 compare also runs Opus |
| Source traceability | Display retrieved chunk/document references with score color and snippets |
| Runtime stats | Show tokens, model latency, tok/sec, retrieval timing |
| Library UX | Dropdown folders by vertical with per-file "Open original document" links |

### 5.3 Security (Hard Requirements)

| Requirement | Description |
|---|---|
| Fully local data plane | Embeddings, chunking, storage, and document files stay local |
| Controlled cloud egress | Only assembled prompts leave machine (Obj 1/2 and optional Obj 3 compare leg) |
| Local sensitive synthesis | Obj 3 default generation is local via Ollama |
| Credential handling | API keys from local env files / environment variables only |
| Repository hygiene | Raw source data folder `/Data/` is excluded from git |

---

## 6. System Prompts

Each objective has a dedicated prompt file:

- `system_prompts/expert_network_brief.md`
  - Brief structure + ~5 screening question guidance
- `system_prompts/interview_guide.md`
  - Stakeholder-specific interview guide structure and sensitive phrasing guidance
- `system_prompts/insights_qa.md`
  - Evidence-first synthesis: counts, quote extraction, who-mentioned-topic, consensus with citations

---

## 7. Technical Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.12 |
| Web framework | FastAPI + Uvicorn |
| Frontend | HTML/CSS/JS single-page app (French-whimsical salon UI) |
| LLM routing | AWS Bedrock Sonnet (Obj 1/2) + Ollama local (Obj 3 default) + optional AWS Bedrock Opus compare |
| Local/cloud model config | `OLLAMA_BASE_URL`, `LOCAL_RAG_MODEL`, `OPUS_COMPARE_MODEL` |
| Embeddings | Sentence Transformers (`all-mpnet-base-v2`) |
| Vector store | ChromaDB (local persistent) |
| Metadata store | SQLite (`uploaded_docs/metadata.db`) |
| Parsing | `python-docx`, `pypdf`, text decoding |
| Local sanitization | LM Studio local API + `sanitize_with_lmstudio.py` |

---

## 8. File Structure

```
Benjamin/
├── backend.py                          # FastAPI server (chat + documents + stats)
├── preprocess.py                       # CLI: ingest/list/remove/stats for RAG data
├── ingest_objective3.py                # CLI: Objective 3-only ingest wrapper (V3 + *.sanitized.txt)
├── sanitize_with_lmstudio.py           # CLI: local de-identification pipeline via LM Studio
├── system_prompts/
│   ├── expert_network_brief.md         # System prompt for Objective 1
│   ├── interview_guide.md              # System prompt for Objective 2
│   └── insights_qa.md                  # System prompt for Objective 3
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── chunking.py
│   ├── document_manager.py
│   ├── retrieval.py
│   └── system_prompts.py
├── static/
│   └── index.html                      # Frontend (form + library + local/opus response panels)
├── chroma_db/                          # Local vector database (gitignored)
├── uploaded_docs/                      # Stored source files + metadata.db (gitignored)
├── Data/                               # Raw working corpus (gitignored)
├── sanitized_data/                     # Sanitized outputs + run report (local)
├── tests/
├── requirements.txt
└── README.md
```

---

## 9. Success Criteria

- **Objective 1**: Produces usable expert-network briefs with ~5 screening questions quickly
- **Objective 2**: Produces stakeholder-tailored interview guides with sensitive-question phrasing
- **Objective 3**: Answers transcript-centric analytical questions with explicit citations and quotes
- **Confidentiality**: Sensitive transcript synthesis can run fully local
- **Compare utility**: Analysts can compare local vs Opus outputs on identical retrieved evidence
- **RAG relevance**: Top retrieved chunks are consistently relevant to the analyst query

---

## 10. Out of Scope (Current Phase)

- Multi-user authentication/authorization
- Cloud deployment of the app
- Cross-session memory beyond current request context
- OCR-heavy handling for scanned/image-only PDFs
- Direct integration with expert-network vendor APIs
