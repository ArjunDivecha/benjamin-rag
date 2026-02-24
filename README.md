# Benjamin PRD — Strategy Consulting AI Assistant

Last updated: 2026-02-22

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
   # optional for live web context in objectives 1/2:
   EXA_API_KEY=your_exa_api_key_here
   # optional for local inference:
   LMSTUDIO_BASE_URL=http://localhost:1234
   LOCAL_RAG_MODEL=qwen2.5:32b
   ```
4. Configure available models in `models.txt` (one per line, `provider|model_id` format):
   ```
   bedrock|us.anthropic.claude-haiku-4-5-20251001-v1:0
   bedrock|us.anthropic.claude-sonnet-4-6
   bedrock|us.anthropic.claude-opus-4-6-v1
   lmstudio|qwen2.5:32b
   lmstudio|qwen3:30b-a3b
   ```
4. Optional (Objective 3 local model): start LM Studio and load a model:
   ```bash
   $HOME/.lmstudio/bin/lms server start
   $HOME/.lmstudio/bin/lms load qwen/qwen3-32b --yes
   ```
5. Put any project files you want to analyze into the `Data/` folder.
6. Start app:
   ```bash
   ./.venv/bin/uvicorn backend:app --port 8000
   ```
   or use the one-command launcher:
   ```bash
   ./run_benjamin.sh
   ```
7. In the UI:
- Click the **Sync Data** (or **Refresh**) button to automatically ingest the files from the `Data/` folder into Benjamin's memory.
- Select any two models from the left/right dropdown bars for side-by-side comparison.
- All three objectives support dual-model comparison across Bedrock and LM Studio providers.
- Optional: check **Add live web context (Exa)** for Brief Salon / Interview Atelier to inject recent web context.
- Per-run metrics include token counts, latency, speed, and estimated cost for Bedrock models.

### Dual Themes

The app ships with two frontend themes you can switch between at any time:

| Route | Theme | Description |
|-------|-------|-------------|
| `http://localhost:8000/` | **Classic** (French salon) | Parchment textures, Parisian bistro styling, Cormorant Garamond + Parisienne fonts |
| `http://localhost:8000/pro` | **Professional** | Clean navy/white consulting aesthetic inspired by benjaminmaurice.com, Inter + Playfair Display fonts |

A toggle link in the top-right corner of each theme switches to the other. Both themes share identical functionality.

---

## TL;DR: Add/Update RAG Data

The easiest way to update Benjamin's knowledge is through the Web UI:
1. Put your real files into the `Data/` folder.
2. Open the Web UI (`http://localhost:8000/` or `http://localhost:8000/ultra`).
3. Click the **Sync Data** (Ultra) or **Refresh** (Classic) button in the side panel.

*Alternatively, you can do this from the terminal:*

```bash
# 1) Go to project
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Benjamin"

# 2) Put your real files into Data/
#    All three objectives share one unified RAG collection.

# 3) Ingest / update RAG (sync removes deleted files too)
./.venv/bin/python preprocess.py --vertical ALL --dir Data --sync

# 4) Verify
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --stats
```

If you edit or add files later, just click **Sync Data** again in the UI.

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
# Optional local-model settings
LMSTUDIO_BASE_URL=http://localhost:1234
LOCAL_RAG_MODEL=qwen2.5:32b
```

### Step 2: Add files into the Data folder

All project files go into `Data/`. There is **one unified RAG collection** that all three objectives (Brief Salon, Interview Atelier, Insights Parlour) share. The system prompts tell each objective how to use the retrieved context, so there is no need to sort files into separate folders.

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

Wait a few seconds for Benjamin to ingest the files. The button will update to tell you how many files were added or removed.

*Alternatively, you can run this command in your terminal:*

```bash
./.venv/bin/python preprocess.py --vertical ALL --dir Data --sync
```

Whether via UI or command line, the sync ensures the RAG is an exact mirror of `Data/`:
- New files are ingested
- Changed files are re-ingested (old chunks replaced)
- Unchanged files are skipped (fast)
- Files you deleted from `Data/` are automatically removed from the RAG

You can also ingest specific files (useful if files are outside `Data/`):

```bash
./.venv/bin/python preprocess.py --vertical ALL --files "/full/path/file1.docx" "/full/path/file2.pdf"
```

### Step 4: Confirm what is in the RAG

```bash
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --stats
```

What to expect in output:
- `ingested (...)` means the file was processed and vectors were updated.
- `skipped (unchanged)` means that exact file content was already ingested in that vertical.
- `skipped (unsupported file type)` means extension is not supported.
- `skipped (read error: ...)` means parsing failed for that file.

### Step 5: How to update data later

Common scenarios:

1. Replace an existing document with a newer version:
   - Keep the same filename in `Data/`.
   - Overwrite the file with new content.
   - Click **Sync Data** in the UI.
   - Result: old chunks for that filename are replaced automatically.

2. Add a brand-new document:
   - Drop it into `Data/`.
   - Click **Sync Data** in the UI.

3. Remove a document from the RAG:
   - Delete it from the `Data/` folder.
   - Click **Sync Data** in the UI. The button will indicate how many files were removed.
   - Remove by `doc_id`:
     ```bash
     ./.venv/bin/python preprocess.py --remove <doc_id>
     ```

---

## Local Data Sanitization (LM Studio, Fully Local)

> **Note:** Sanitization is optional. If you are comfortable with raw data being used in prompts sent to Bedrock, you can skip this section entirely and ingest raw files directly from `Data/`.

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

### Build RAG from sanitized output (optional)

After sanitization finishes, you can ingest the sanitized files instead of raw data:

```bash
./.venv/bin/python preprocess.py --vertical ALL --dir sanitized_data
./.venv/bin/python preprocess.py --list
./.venv/bin/python preprocess.py --stats
```

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

**Client and company confidentiality is non-negotiable.** The current implementation uses a flexible dual-model comparison architecture:

- All three objectives support side-by-side model comparison via configurable `models.txt`.
- Models from **AWS Bedrock** (Claude Haiku, Sonnet, Opus) and **LM Studio** (local models) can be freely mixed.
- Users select left/right models from dropdown bars in the UI; the backend dynamically routes to the appropriate provider.

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

A local preprocessing step ingests source documents and builds a unified RAG index:

- **Input**: All project documents — briefs, interview guides, interview notes, presentations, reports, and related research artifacts
- **Processing**: Chunk -> embed (Sentence Transformers, MPS-accelerated) -> store in Chroma
- **Output**: One unified collection (`ALL`) shared by all three objectives

All three objectives retrieve from the same collection. The objective-specific system prompts (see Section 6) tell the LLM how to use the retrieved context for each task.

The pipeline supports incremental updates. Re-ingesting an updated file replaces stale chunks for that filename.

### 4.2 Runtime Routing (Current)

Model routing is now fully dynamic. Available models are defined in `models.txt` (format: `provider|model_id`, one per line). The UI presents two dropdown selectors for side-by-side comparison. Every request goes to `/api/chat/compare` with user-selected `model_left` and `model_right`.

| Provider | Example Models | Cloud Egress |
|---|---|---|
| `bedrock` | `us.anthropic.claude-haiku-4-5-20251001-v1:0`, `us.anthropic.claude-sonnet-4-6`, `us.anthropic.claude-opus-4-6-v1` | Yes |
| `lmstudio` | `qwen2.5:32b`, `qwen3:30b-a3b` | No |

**Estimated cost per request** is calculated and displayed in the UI for Bedrock models:

| Model Family | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.6 | $5.00 | $25.00 |
| LM Studio (local) | $0.00 | $0.00 |

### 4.3 Web Application (Runtime)

A FastAPI + HTML frontend where consultants interact with three sections (Brief Salon, Interview Atelier, Insights Parlour):

1. Select objective
2. Choose mode
   - `direct`: optional file context + prompt
   - `rag`: retrieve from V1/V2/V3 + objective-specific prompt
3. Select left and right models from dropdown bars (populated from `models.txt` via `/api/models`)
4. Optional: enable live web context (Exa) for objectives 1/2
5. Retrieve relevant chunks from the mapped vertical
6. Load objective-specific prompt from `system_prompts/*.md`
7. Route both models to their respective providers in parallel
8. Display answers side-by-side with per-model metrics (tokens, latency, speed, estimated cost)

Current UX behavior:
- Both response panels are always active across all three objectives
- Model dropdowns default to the first two models in `models.txt`
- Metrics are displayed as compact pills at the top of each response card
- Source cards are color-coded by confidence and include snippets

### 4.4 API Surface (Current)

- `POST /api/chat`
  - `mode=direct`: optional file context + prompt, AWS Bedrock generation
  - `mode=rag`: requires `objective` in `{expert_network_brief, interview_guide, insights_qa}`
  - Optional `use_web_search=true` runs Exa retrieval and injects `<web_context>` for objectives 1/2
  - RAG response includes `content`, `usage`, `provider`, `model`, `metrics`, `rag`, `sources`
- `POST /api/chat/compare`
  - Accepts any objective and any pair of models
  - Parameters: `message`, `mode`, `objective`, `model_left`, `model_right`, `top_k`, `min_score`, `file`, `use_web_search`
  - `model_left` and `model_right` use `provider|model_id` format (e.g., `bedrock|us.anthropic.claude-sonnet-4-6`)
  - Both `direct` and `rag` modes are supported
  - Returns `left`, `right`, shared `rag`, shared `sources`, shared `web_search`, and request-level `metrics`
- `GET /api/models`
  - Returns available models parsed from `models.txt` (used by frontend dropdowns)
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
│ LM Studio local model (Obj 3)│          
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
| Vertical assignment | All files go into the unified `ALL` collection shared by all objectives |
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
| Mode selection | Direct upload mode and RAG mode |
| RAG retrieval | Query the unified `ALL` collection with `top_k` and `min_score` |
| Objective prompts | Loads objective-specific system prompts from local files |
| Dual-model comparison | Any two models (Bedrock or LM Studio) can be compared side-by-side on any objective |
| Dynamic model selection | Models are loaded from `models.txt` and presented via dropdown selectors |
| Source traceability | Display retrieved chunk/document references with score color and snippets |
| Runtime stats | Show tokens, latency (seconds), speed (tok/s), and estimated cost per run |
| Library UX | Dropdown folders by vertical with per-file "Open original document" links |

### 5.3 Security (Hard Requirements)

| Requirement | Description |
|---|---|
| Fully local data plane | Embeddings, chunking, storage, and document files stay local |
| Controlled cloud egress | Only assembled prompts leave machine (Obj 1/2 and optional Obj 3 compare leg) |
| Local sensitive synthesis | Obj 3 default generation is local via LM Studio |
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
| LLM routing | Dynamic dual-model comparison via `models.txt` — supports AWS Bedrock (Claude Haiku/Sonnet/Opus) and LM Studio (local models) |
| Model configuration | `models.txt` (provider\|model_id per line), `/api/models` endpoint |
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
├── sanitize_with_lmstudio.py           # CLI: local de-identification pipeline via LM Studio (optional)
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
│   ├── index.html                      # Frontend — Classic French theme (form + library + dual-model response panels)
│   ├── index_pro.html                  # Frontend — Professional theme (same functionality, clean navy/white design)
│   ├── pro-theme.css                   # CSS for professional theme
│   └── app.js                          # Shared application JavaScript (used by professional theme)
├── models.txt                          # Available models config (provider|model_id per line)
├── chroma_db/                          # Local vector database (gitignored)
├── uploaded_docs/                      # Stored source files + metadata.db (gitignored)
├── Data/                               # Raw working corpus — all project files go here (gitignored)
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
- **Compare utility**: Analysts can compare any two models (Bedrock or local) side-by-side on identical retrieved evidence, with per-run cost shown
- **RAG relevance**: Top retrieved chunks are consistently relevant to the analyst query

---

## 10. Out of Scope (Current Phase)

- Multi-user authentication/authorization
- Cloud deployment of the app
- Cross-session memory beyond current request context
- OCR-heavy handling for scanned/image-only PDFs
- Direct integration with expert-network vendor APIs
