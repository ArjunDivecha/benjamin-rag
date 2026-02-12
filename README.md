# Benjamin PRD — Strategy Consulting AI Assistant

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
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
4. Optional: ingest synthetic starter corpus:
   ```bash
   ./.venv/bin/python preprocess.py --vertical V1 --dir synthetic_data/V1
   ./.venv/bin/python preprocess.py --vertical V2 --dir synthetic_data/V2
   ```
5. Start app:
   ```bash
   ./.venv/bin/uvicorn backend:app --port 8000
   ```

---

## 1. Product Vision

Benjamin is an internal AI tool for a boutique strategy consulting firm that competes with McKinsey, BCG, and Bain. It accelerates repetitive components of client engagements — specifically **expert network briefs** and **interview guide drafting** — by combining curated knowledge bases (RAGs) with Claude Sonnet 4.5 and domain-specific system prompts.

**Client and company confidentiality is non-negotiable.** All processing outside the single OpenRouter API call (routed through Amazon Bedrock with Zero Data Retention) runs entirely on a local M2 Mac. No data touches the cloud or web beyond that one call.

---

## 2. Users & Context

- **Users**: Consultants at the firm conducting commercial due diligence and growth strategy engagements
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
- Include ~5 screening questions (±1) that:
  - Cover a representative set of interview topics
  - Assess interviewees' level of knowledge
  - Extract useful information even if the person is not ultimately interviewed
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

---

## 4. System Architecture

### 4.1 Preprocessing Pipeline (Offline / Batch)

A local preprocessing step ingests source documents and builds **multiple RAG indices across different verticals**:

- **Input**: Multiple files (past briefs, interview guides, methodology docs, industry templates, best practices, etc.)
- **Processing**: Chunk → embed (Sentence Transformers, MPS-accelerated) → store in Chroma
- **Output**: Two named RAG collections (verticals):
  - **V1** — maps to Objective 1 (expert network briefs, screening question examples, past project briefs)
  - **V2** — maps to Objective 2 (interview guides by stakeholder type, question design patterns, methodology)

Each vertical is a separate Chroma collection with its own metadata schema, allowing targeted retrieval per objective.

The pipeline is designed for **easy incremental updates** — as new project briefs, interview guides, or methodology documents are completed, they can be added to the relevant vertical without rebuilding the entire index. Updated or replaced documents are automatically re-chunked and re-embedded, with stale chunks removed.

### 4.2 Web Application (Runtime)

A FastAPI + HTML frontend where consultants interact with the system:

1. **Select objective** (brief drafting or interview guide)
2. **Provide project context** (objectives, questions, stakeholder type, industry)
3. **System retrieves** relevant chunks from the corresponding vertical (V1 or V2)
4. **Custom system prompt** (per objective) instructs Claude on firm style, output format, and quality standards
5. **Claude Sonnet 4.5** generates the output via OpenRouter → Bedrock + ZDR
6. **Response displayed** with source citations from the RAG

### 4.3 Security Model

```
LOCAL (M2 Mac - 24GB)                    CLOUD (only network call)
┌─────────────────────────────┐          ┌──────────────────────────┐
│ All source documents        │          │ OpenRouter API           │
│ All RAG indices (Chroma)    │          │  → Amazon Bedrock        │
│ All embeddings              │   ──►    │  → ZDR enforced          │
│ All preprocessing           │ (prompt  │  → No data retention     │
│ Web UI + FastAPI backend    │  only)   │  → No fallback providers │
│ System prompts              │          └──────────────────────────┘
│ User inputs                 │
└─────────────────────────────┘
```

- **Zero leakage**: No document content, embeddings, or metadata leaves the local machine
- **Only the assembled prompt** (retrieved chunks + user query + system prompt) is sent to OpenRouter
- **ZDR**: OpenRouter routes exclusively to Bedrock endpoints with zero data retention
- **No fallbacks**: `allow_fallbacks: false` ensures no silent rerouting to non-ZDR providers

---

## 5. Functional Requirements

### 5.1 Preprocessing CLI

| Requirement | Description |
|---|---|
| Multi-file ingestion | Accept a directory or list of files for batch processing |
| Vertical assignment | Each file is tagged to a vertical (V1 or V2) |
| Chunking | 512-token chunks with 50-token overlap |
| Local embeddings | Sentence Transformers (`all-mpnet-base-v2`), MPS-accelerated |
| Persistent storage | Chroma with DuckDB backend, one collection per vertical |
| Metadata tracking | SQLite for document metadata (filename, vertical, chunk count, ingestion date) |
| Idempotent re-ingestion | Re-ingesting a file replaces its chunks (content-hash-based doc IDs) |
| Incremental updates | Add new files to an existing vertical without rebuilding the full index |
| Single-file add/remove | Add or remove individual documents from a vertical with one command |
| Update detection | Skip files that haven't changed since last ingestion (content-hash comparison) |

### 5.2 Web Application

| Requirement | Description |
|---|---|
| Objective selection | User picks Objective 1 (brief) or Objective 2 (interview guide) |
| Context input | Free-text fields for project objectives, key questions, industry, stakeholder type |
| RAG retrieval | Queries the corresponding vertical (V1 for Obj 1, V2 for Obj 2) |
| Custom system prompts | Per-objective prompts encoding firm methodology and output format |
| LLM generation | Claude Sonnet 4.5 via OpenRouter (Bedrock + ZDR, no fallbacks) |
| Source citations | Display which RAG chunks informed the response |
| Document library | View/manage ingested documents and verticals |

### 5.3 Security (Hard Requirements)

| Requirement | Description |
|---|---|
| Fully local processing | Embeddings, chunking, storage, UI — all on local machine |
| No cloud storage | No S3, no cloud databases, no external vector stores |
| No telemetry | No analytics, no usage tracking to external services |
| ZDR enforcement | `provider.zdr: true` on every OpenRouter call |
| Bedrock-only routing | `provider.order: ["amazon-bedrock"]`, `allow_fallbacks: false` |
| Credential security | API keys via 1Password or local env file only |

---

## 6. System Prompts (Suggested)

Each objective uses a dedicated system prompt stored as a local markdown file. Below are suggested starting points — these should be iterated based on output quality.

### 6.1 Objective 1: Expert Network Brief (`system_prompts/expert_network_brief.md`)

> You are an expert research assistant at a boutique strategy consulting firm. Your task is to draft a project brief for an expert network (e.g., GLG, AlphaSights) to help identify and screen relevant stakeholders for interviews.
>
> **Output format:**
> 1. **Project Background** — 2-3 sentence summary of the engagement context
> 2. **Target Expert Profile** — Description of ideal interviewees (role, industry, experience level)
> 3. **Screening Questions** — Exactly 5 questions (±1) that:
>    - Cover a representative cross-section of the topics to be discussed in actual interviews
>    - Assess the interviewee's depth of knowledge on the subject matter
>    - Yield useful data points even if the person is not ultimately selected for a full interview
>    - Are concise and answerable in 1-2 sentences
>
> **Style guidelines:**
> - Professional, concise, and specific to the industry/market in question
> - Avoid jargon that expert network coordinators may not understand
> - Screening questions should be closed-ended or short-answer where possible
>
> Use the provided reference materials (past briefs, methodology) to match the firm's established format and quality bar.

### 6.2 Objective 2: Interview Guide (`system_prompts/interview_guide.md`)

> You are an expert research assistant at a boutique strategy consulting firm. Your task is to draft an interview guide — a structured list of questions for a specific stakeholder type.
>
> **Output format:**
> 1. **Interview Context** — Brief description of the engagement and this interview's role in it
> 2. **Stakeholder Type** — Who is being interviewed (e.g., customer, competitor, internal employee)
> 3. **Questions** — Organized by topic area, with:
>    - Opening questions (rapport-building, broad context)
>    - Core questions (directly addressing project objectives)
>    - Sensitive questions (phrased to maximize candor without revealing intent)
>    - Closing questions (catch-all, referrals)
>
> **Question design principles:**
> - Optimize for usefulness of responses — ask questions that yield actionable data
> - Optimize for willingness to speak — phrase sensitive topics indirectly when needed
> - For competitive intelligence: frame questions around "market trends" or "industry practices" rather than asking directly about a competitor's strategy
> - Tailor language and depth to the stakeholder type (e.g., C-suite vs. operational staff)
>
> Use the provided reference materials (past guides, methodology) to match the firm's established format and quality bar.

---

## 7. Technical Stack

| Component | Technology |
|---|---|
| **Runtime** | Python 3.11+, M2 Mac (24GB) |
| **Web framework** | FastAPI + Uvicorn |
| **Frontend** | HTML/CSS/JS (single-page, served by FastAPI) |
| **LLM** | Claude Sonnet 4.5 via OpenRouter → Amazon Bedrock (ZDR) |
| **Embeddings** | Sentence Transformers (`all-mpnet-base-v2`), MPS device |
| **Vector store** | ChromaDB (persistent, local) |
| **Metadata store** | SQLite |
| **Document parsing** | python-docx, pypdf, plain text |
| **Credentials** | `OPENROUTER_API_KEY` via local `.env` or environment variable |

---

## 8. File Structure (Target)

```
Benjamin/
├── backend.py                          # FastAPI server (chat + RAG endpoints)
├── preprocess.py                       # CLI: ingest files → chunk → embed → store
├── system_prompts/
│   ├── expert_network_brief.md         # System prompt for Objective 1
│   └── interview_guide.md              # System prompt for Objective 2
├── rag/
│   ├── __init__.py
│   ├── embeddings.py                   # Local embedding model wrapper
│   ├── vector_store.py                 # Chroma client (multi-collection)
│   ├── chunking.py                     # Text chunking strategies
│   ├── document_manager.py             # Document metadata CRUD (SQLite)
│   └── retrieval.py                    # Query → retrieve → assemble context
├── static/
│   └── index.html                      # Web frontend
├── chroma_db/                          # Vector database (local, gitignored)
├── uploaded_docs/                      # Original documents (local, gitignored)
│   └── metadata.db                     # SQLite metadata
├── requirements.txt                    # Python dependencies
├── README.md                           # This PRD
└── RAG_IMPLEMENTATION_PLAN.md          # Detailed technical implementation plan
```

---

## 9. Success Criteria

- **Objective 1**: Given project objectives, the system produces a usable expert network brief with ~5 screening questions in under 30 seconds
- **Objective 2**: Given project objectives and stakeholder type, the system produces a tailored interview guide with appropriately phrased questions in under 30 seconds
- **Confidentiality**: Zero data leakage — verified by network audit (only OpenRouter calls leave the machine)
- **Quality**: Output quality comparable to manually drafted briefs/guides, requiring minimal editing
- **RAG relevance**: Retrieved chunks are demonstrably relevant (>80% precision in top-5)

---

## 10. Out of Scope (Current Phase)

- Multi-user access / authentication
- Cloud deployment
- Conversation memory across sessions
- PDF table/image extraction
- Real-time collaborative editing
- Integration with expert network platforms (e.g., GLG, AlphaSights APIs)
