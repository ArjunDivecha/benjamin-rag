# Detailed Plan: Local Vector RAG for Document Chat (M2 24GB)

> Status note (2026-02-17): This is a historical deep-design plan.
> The implemented system has evolved from this draft (notably Objective 3 local + Opus compare flow and current API/UI shape).
> For current behavior use:
> - `README.md`
> - `docs/IMPLEMENTATION_STATUS.md`
> - `docs/REPO_CONTEXT.md`

## Project Overview

Extend the existing Sonnet + File Context web app to support vector RAG for large documents, using patterns from your Memory system but fully local for security.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    M2 Mac (24GB) - ALL LOCAL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Web UI (React/HTML)                                            │
│    ├── Upload documents                                         │
│    ├── View document library                                    │
│    ├── Chat with documents                                      │
│    └── Delete/manage documents                                  │
│                                                                  │
│  FastAPI Backend                                                │
│    ├── POST /api/ingest       (upload → chunk → embed → store) │
│    ├── POST /api/chat         (query → retrieve → LLM)         │
│    ├── GET  /api/documents    (list all documents)             │
│    ├── DELETE /api/documents/:id                               │
│    └── GET  /api/stats        (storage stats)                  │
│                                                                  │
│  Embedding Layer (Sentence Transformers)                        │
│    ├── Model: all-mpnet-base-v2 (768 dim)                     │
│    ├── Device: MPS (Metal GPU acceleration)                    │
│    ├── Batch size: 32 chunks                                   │
│    └── Memory: ~420MB                                           │
│                                                                  │
│  Vector Store (Chroma)                                          │
│    ├── Backend: DuckDB (persistent)                            │
│    ├── Path: ./chroma_db/                                      │
│    ├── Collections: documents, metadata                        │
│    └── Metadata: filename, chunk_id, doc_id, uploaded_at       │
│                                                                  │
│  Document Storage (Local filesystem)                            │
│    ├── Path: ./uploaded_docs/                                  │
│    ├── Format: original files preserved                        │
│    └── Index: SQLite for metadata                              │
│                                                                  │
│  LLM (OpenRouter → Bedrock + ZDR)                              │
│    └── Only network call - retrieved context sent              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Benjamin/
├── backend.py                    # Main FastAPI app (extended)
├── static/
│   └── index.html               # Web UI (extended with RAG features)
├── rag/
│   ├── __init__.py
│   ├── embeddings.py            # Sentence Transformers wrapper (from Memory pattern)
│   ├── vector_store.py          # Chroma client (adapted from Memory vector_client.py)
│   ├── chunking.py              # Text chunking strategies
│   ├── document_manager.py      # Document CRUD + metadata
│   └── retrieval.py             # Query → retrieve → rerank logic
├── chroma_db/                   # Vector database (created on first run)
├── uploaded_docs/               # Original documents
│   └── metadata.db             # SQLite: doc metadata
├── .venv/                       # Python virtual environment
├── requirements.txt             # Updated with new deps
├── README.md                    # Updated documentation
└── RAG_IMPLEMENTATION_PLAN.md   # This document
```

---

## Phase 1: Core Infrastructure

### 1.1 Embedding Layer (`rag/embeddings.py`)

**Purpose:** Local embedding generation using Sentence Transformers

**Key Components:**
- Initialize `all-mpnet-base-v2` with MPS device
- Singleton pattern for model (load once, reuse)
- Batch processing (32 chunks at a time for M2)
- Token counting (approximate for local model)
- Memory management (cleanup after large batches)

**API:**
```python
def get_embedding(text: str) -> list[float]
def get_embeddings_batch(texts: list[str]) -> list[list[float]]
def estimate_tokens(text: str) -> int
def cleanup_model_cache()
```

**Memory optimization:**
- Lazy load model (only when first needed)
- Clear cache after large operations
- Use FP16 if memory constrained

**Pattern from Memory system:**
```python
# From Memory/knowledge-system/distillation/utils/embedding.py
# Adapt the batch processing and client singleton pattern
def get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# Becomes:
def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer('all-mpnet-base-v2', device='mps')
    return _model
```

---

### 1.2 Vector Store (`rag/vector_store.py`)

**Purpose:** Chroma wrapper with same interface as Memory system's Upstash client

**Key Components:**
- PersistentClient with local path
- Collection management (one per document or unified)
- Metadata schema: `{doc_id, filename, chunk_id, uploaded_at, file_type}`
- Batch upsert (100 vectors max per call)
- Query with filtering and min_score

**API:**
```python
class VectorStore:
    def __init__(self, persist_path: str)
    def upsert_document(doc_id, chunks, embeddings, metadata)
    def search(query_embedding, top_k, filter_metadata, min_score)
    def delete_document(doc_id)
    def get_stats() -> dict  # vector count, size, etc.
    def test_connection() -> tuple[bool, str]
```

**Metadata schema:**
```python
{
    "doc_id": "doc_abc123",
    "filename": "russia_analysis.docx",
    "chunk_id": 5,
    "uploaded_at": "2026-02-12T00:00:00Z",
    "file_type": "docx",
    "total_chunks": 42
}
```

**Pattern from Memory system:**
```python
# From Memory/knowledge-system/distillation/storage/vector_client.py
# Keep the same interface, swap Upstash for Chroma

class VectorClient:
    def __init__(self):
        self.index = Index(
            url=UPSTASH_VECTOR_REST_URL,
            token=UPSTASH_VECTOR_REST_TOKEN,
        )
    
    def upsert_entry(self, entry_id, vector, metadata):
        self.index.upsert(vectors=[{
            "id": entry_id,
            "vector": vector,
            "metadata": metadata
        }])

# Becomes:
class VectorStore:
    def __init__(self, persist_path: str):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("documents")
    
    def upsert_entry(self, entry_id, vector, metadata):
        self.collection.add(
            ids=[entry_id],
            embeddings=[vector],
            metadatas=[metadata]
        )
```

---

### 1.3 Chunking Strategy (`rag/chunking.py`)

**Purpose:** Split documents into optimal chunks for embedding

**Strategies:**
1. **Fixed size** (default): 512 tokens, 50 token overlap
2. **Semantic**: Split by paragraphs, merge to target size
3. **Recursive**: Sentence-based with smart merging

**API:**
```python
def chunk_text(text: str, strategy: str = "fixed") -> list[str]
def chunk_with_metadata(text: str) -> list[dict]  # chunk + position info
def estimate_chunk_count(text: str) -> int
```

**Configuration:**
```python
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap
MIN_CHUNK_SIZE = 100      # discard smaller chunks
MAX_CHUNK_SIZE = 1000     # split larger chunks
```

**Implementation approach:**
```python
import tiktoken

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Target tokens per chunk
        overlap: Tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    
    return chunks
```

---

### 1.4 Document Manager (`rag/document_manager.py`)

**Purpose:** Manage document lifecycle and metadata

**Key Components:**
- SQLite database for document metadata
- File storage in `uploaded_docs/`
- Document ID generation (hash-based or UUID)
- CRUD operations
- Storage stats tracking

**Database Schema:**
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0
);
```

**API:**
```python
class DocumentManager:
    def save_document(file_bytes, filename) -> str  # returns doc_id
    def get_document(doc_id) -> dict
    def list_documents() -> list[dict]
    def delete_document(doc_id)
    def update_access(doc_id)
    def get_storage_stats() -> dict
```

**Implementation:**
```python
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

class DocumentManager:
    def __init__(self, storage_path: str = "./uploaded_docs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.db_path = self.storage_path / "metadata.db"
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def save_document(self, file_bytes: bytes, filename: str) -> str:
        # Generate doc_id from content hash
        doc_id = "doc_" + hashlib.sha256(file_bytes).hexdigest()[:12]
        
        # Save file
        file_path = self.storage_path / f"{doc_id}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        return doc_id
```

---

### 1.5 Retrieval Logic (`rag/retrieval.py`)

**Purpose:** Query → retrieve relevant chunks → optionally rerank

**Key Components:**
- Embed query using same model
- Vector search (top_k = 10-20 initially)
- Optional reranking (by recency, source diversity)
- Context assembly (combine chunks with metadata)
- Deduplication (if same chunk appears multiple times)

**API:**
```python
def retrieve_context(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] = None,  # filter by specific docs
    min_score: float = 0.5
) -> list[dict]  # [{chunk, score, metadata}]

def assemble_context(chunks: list[dict]) -> str  # format for LLM
def rerank_by_recency(chunks: list[dict]) -> list[dict]
```

**Context formatting:**
```
<document name="russia_analysis.docx" chunk="5/42">
[chunk text here]
</document>

<document name="market_report.pdf" chunk="12/89">
[chunk text here]
</document>
```

**Implementation:**
```python
def retrieve_context(
    query: str,
    vector_store: VectorStore,
    embedding_model,
    top_k: int = 5,
    doc_ids: list[str] = None,
    min_score: float = 0.5
) -> list[dict]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: User question
        vector_store: VectorStore instance
        embedding_model: Embedding model
        top_k: Number of chunks to retrieve
        doc_ids: Optional list of doc_ids to filter
        min_score: Minimum similarity score
    
    Returns:
        List of chunks with metadata
    """
    # Embed query
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    # Search
    filter_metadata = None
    if doc_ids:
        filter_metadata = {"doc_id": {"$in": doc_ids}}
    
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_metadata=filter_metadata,
        min_score=min_score
    )
    
    return results

def assemble_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for LLM context."""
    context_parts = []
    for chunk in chunks:
        meta = chunk['metadata']
        context_parts.append(
            f'<document name="{meta["filename"]}" '
            f'chunk="{meta["chunk_id"]}/{meta["total_chunks"]}">\n'
            f'{chunk["text"]}\n'
            f'</document>'
        )
    return "\n\n".join(context_parts)
```

---

## Phase 2: Backend API Extensions

### 2.1 Ingestion Endpoint

**Route:** `POST /api/ingest`

**Request:**
- Multipart form: `file` (docx, txt, pdf, md, etc.)
- Optional: `doc_id` (for updates)

**Process:**
1. Validate file (size, type)
2. Extract text (existing `_read_file_content` + PDF support)
3. Generate doc_id (or use provided)
4. Save original file to `uploaded_docs/`
5. Chunk text (512 tokens, 50 overlap)
6. Batch embed chunks (32 at a time)
7. Upsert to vector store with metadata
8. Save metadata to SQLite
9. Return doc_id + stats

**Response:**
```json
{
  "doc_id": "doc_abc123",
  "filename": "russia_analysis.docx",
  "chunks": 42,
  "tokens": 21500,
  "embedding_time_ms": 3200,
  "storage_size_mb": 0.016
}
```

**Error handling:**
- File too large (>50MB)
- Unsupported format
- Embedding failure
- Storage full

**Implementation:**
```python
@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document: extract → chunk → embed → store.
    """
    start_time = time.time()
    
    # Read file
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Extract text
    text = _read_file_content(file_bytes, file.filename)
    
    # Generate doc_id
    doc_id = doc_manager.save_document(file_bytes, file.filename)
    
    # Chunk
    chunks = chunk_text(text)
    
    # Embed (batch)
    embeddings = embedding_model.encode(chunks, batch_size=32)
    
    # Store
    vector_store.upsert_document(
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings.tolist(),
        metadata={
            "filename": file.filename,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    )
    
    # Save metadata
    doc_manager.save_metadata(doc_id, file.filename, len(chunks))
    
    elapsed = (time.time() - start_time) * 1000
    
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "embedding_time_ms": elapsed
    }
```

---

### 2.2 Chat Endpoint (Extended)

**Route:** `POST /api/chat`

**Request:**
```json
{
  "message": "What did the Russia analysis say about sanctions?",
  "mode": "rag",  // or "direct" for current behavior
  "doc_ids": ["doc_abc123"],  // optional: filter to specific docs
  "top_k": 5
}
```

**Process:**
1. If `mode == "direct"`: use existing file upload logic
2. If `mode == "rag"`:
   - Embed query
   - Retrieve top_k chunks from vector store
   - Filter by doc_ids if provided
   - Assemble context
   - Send to OpenRouter with context + query
3. Track usage (tokens, cost)

**Response:**
```json
{
  "content": "According to the analysis...",
  "usage": {
    "prompt_tokens": 1850,
    "completion_tokens": 120
  },
  "sources": [
    {"doc_id": "doc_abc123", "chunk": 5, "score": 0.87},
    {"doc_id": "doc_abc123", "chunk": 12, "score": 0.82}
  ]
}
```

**Implementation:**
```python
@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    mode: str = Form("rag"),
    doc_ids: Optional[str] = Form(None),  # JSON string
    top_k: int = Form(5),
    file: Optional[UploadFile] = File(None)
):
    """
    Chat with documents (RAG mode) or direct file upload.
    """
    if mode == "direct" and file:
        # Existing logic: read file, send to LLM
        pass
    
    elif mode == "rag":
        # Parse doc_ids
        doc_id_list = json.loads(doc_ids) if doc_ids else None
        
        # Retrieve context
        chunks = retrieve_context(
            query=message,
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=top_k,
            doc_ids=doc_id_list
        )
        
        # Assemble context
        context = assemble_context(chunks)
        
        # Send to LLM
        full_prompt = f"{context}\n\nUser question: {message}"
        
        # Call OpenRouter (existing logic)
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 4096,
                "provider": {
                    "order": ["amazon-bedrock"],
                    "zdr": True,
                    "allow_fallbacks": False
                }
            }
        )
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return {
            "content": content,
            "usage": data["usage"],
            "sources": [
                {
                    "doc_id": c["metadata"]["doc_id"],
                    "chunk": c["metadata"]["chunk_id"],
                    "score": c["score"]
                }
                for c in chunks
            ]
        }
```

---

### 2.3 Document Management Endpoints

**List documents:**
```
GET /api/documents
Response: [{doc_id, filename, chunks, uploaded_at, size_mb}]
```

**Get document details:**
```
GET /api/documents/:doc_id
Response: {doc_id, filename, chunks, metadata, access_count}
```

**Delete document:**
```
DELETE /api/documents/:doc_id
Process:
  1. Delete from vector store
  2. Delete from SQLite
  3. Delete original file
Response: {success: true}
```

**Storage stats:**
```
GET /api/stats
Response: {
  total_documents: 42,
  total_chunks: 8500,
  total_size_mb: 125,
  vector_db_size_mb: 45,
  model_loaded: true,
  model_memory_mb: 420
}
```

**Implementation:**
```python
@app.get("/api/documents")
def list_documents():
    """List all documents."""
    return doc_manager.list_documents()

@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    """Get document details."""
    doc = doc_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete document and all associated data."""
    # Delete from vector store
    vector_store.delete_document(doc_id)
    
    # Delete from document manager
    doc_manager.delete_document(doc_id)
    
    return {"success": True}

@app.get("/api/stats")
def get_stats():
    """Get storage and system stats."""
    return {
        "total_documents": doc_manager.count_documents(),
        "total_chunks": vector_store.count_vectors(),
        "vector_db_size_mb": vector_store.get_size_mb(),
        "model_loaded": embedding_model is not None,
    }
```

---

## Phase 3: Frontend Enhancements

### 3.1 Document Library View

**New section in UI:**
- Table/list of uploaded documents
- Columns: filename, size, chunks, uploaded date, actions
- Actions: view, delete, chat with this doc
- Upload button (triggers ingestion)
- Progress indicator during upload/embedding

**HTML structure:**
```html
<div id="documentLibrary">
  <h2>Document Library</h2>
  <button id="uploadDoc">Upload Document</button>
  <table id="docTable">
    <thead>
      <tr>
        <th>Name</th>
        <th>Size</th>
        <th>Chunks</th>
        <th>Date</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody>
      <!-- Populated via JS -->
    </tbody>
  </table>
</div>
```

**JavaScript:**
```javascript
async function loadDocuments() {
  const res = await fetch('/api/documents');
  const docs = await res.json();
  
  const tbody = document.querySelector('#docTable tbody');
  tbody.innerHTML = docs.map(doc => `
    <tr>
      <td>${doc.filename}</td>
      <td>${(doc.size_mb).toFixed(2)} MB</td>
      <td>${doc.chunks}</td>
      <td>${new Date(doc.uploaded_at).toLocaleDateString()}</td>
      <td>
        <button onclick="chatWithDoc('${doc.doc_id}')">Chat</button>
        <button onclick="deleteDoc('${doc.doc_id}')">Delete</button>
      </td>
    </tr>
  `).join('');
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await fetch('/api/ingest', {
    method: 'POST',
    body: formData
  });
  
  const result = await res.json();
  alert(`Uploaded: ${result.chunks} chunks in ${result.embedding_time_ms}ms`);
  loadDocuments();
}
```

---

### 3.2 Chat Mode Toggle

**UI changes:**
- Radio buttons: "Direct mode" vs "RAG mode"
- In RAG mode:
  - Show document selector (multi-select)
  - Show top_k slider (3-10)
  - Show min_score slider (0.5-0.9)
- In Direct mode:
  - Current file upload behavior

**HTML:**
```html
<div class="mode-selector">
  <label>
    <input type="radio" name="mode" value="direct" checked>
    Direct Mode (upload file)
  </label>
  <label>
    <input type="radio" name="mode" value="rag">
    RAG Mode (search documents)
  </label>
</div>

<div id="ragOptions" style="display:none">
  <label>Documents:</label>
  <select id="docSelector" multiple>
    <!-- Populated from /api/documents -->
  </select>
  
  <label>Top K: <span id="topKValue">5</span></label>
  <input type="range" id="topK" min="3" max="10" value="5">
</div>
```

---

### 3.3 Source Citations

**Display retrieved chunks:**
- Show which documents/chunks were used
- Similarity scores
- Click to view full chunk
- Highlight relevant sections

**Example:**
```
Response: [model answer]

Sources:
📄 russia_analysis.docx (chunk 5/42) - score: 0.87
📄 russia_analysis.docx (chunk 12/42) - score: 0.82
📄 market_report.pdf (chunk 23/156) - score: 0.79
```

**HTML:**
```html
<div class="sources">
  <h4>Sources</h4>
  <ul id="sourceList">
    <!-- Populated via JS -->
  </ul>
</div>
```

**JavaScript:**
```javascript
function displaySources(sources) {
  const list = document.getElementById('sourceList');
  list.innerHTML = sources.map(s => `
    <li>
      📄 ${s.filename} (chunk ${s.chunk}) - score: ${s.score.toFixed(2)}
    </li>
  `).join('');
}
```

---

## Phase 4: Optimization & Polish

### 4.1 Performance Optimizations

**Embedding:**
- Preload model on startup (don't lazy load)
- Use FP16 for faster inference
- Batch size tuning based on available memory
- Cache frequently accessed embeddings

```python
# Startup optimization
@app.on_event("startup")
async def startup_event():
    global embedding_model, vector_store, doc_manager
    
    # Preload model
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device='mps')
    
    # Initialize stores
    vector_store = VectorStore(persist_path="./chroma_db")
    doc_manager = DocumentManager(storage_path="./uploaded_docs")
    
    print("✅ Model loaded, vector store ready")
```

**Vector search:**
- Index optimization (Chroma HNSW parameters)
- Metadata indexing for fast filtering
- Result caching for repeated queries

**Memory management:**
- Monitor memory usage
- Periodic garbage collection
- Clear model cache if memory > 20GB
- Warn user if approaching limits

```python
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    if mem.percent > 85:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

---

### 4.2 Advanced Features (Optional)

**Hybrid search:**
- Combine vector search with keyword search (BM25)
- Weight: 70% semantic, 30% keyword

**Reranking:**
- Use cross-encoder for better relevance
- Or simple heuristics (recency, source diversity)

**Multi-document chat:**
- Retrieve from multiple documents
- Cite sources clearly
- Handle contradictions

**Conversation memory:**
- Store chat history
- Use previous context for follow-up questions

---

### 4.3 Configuration File

**`rag/config.py`:**
```python
# Embedding
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DEVICE = "mps"  # or "cpu"
EMBEDDING_BATCH_SIZE = 32

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100

# Vector Store
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# Retrieval
DEFAULT_TOP_K = 5
MIN_SIMILARITY_SCORE = 0.5
MAX_CONTEXT_TOKENS = 3000

# Storage
UPLOADED_DOCS_PATH = "./uploaded_docs"
METADATA_DB_PATH = "./uploaded_docs/metadata.db"
MAX_FILE_SIZE_MB = 50
MAX_TOTAL_STORAGE_GB = 10

# OpenRouter (existing)
MODEL = "anthropic/claude-sonnet-4.5"
PROVIDER = "amazon-bedrock"
ZDR = True
```

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

**Test coverage:**
- Embedding generation (single + batch)
- Chunking strategies (fixed, semantic)
- Vector store operations (upsert, query, delete)
- Document manager CRUD
- Retrieval accuracy

**Test data:**
- Small doc (1 page, ~5 chunks)
- Medium doc (20 pages, ~100 chunks)
- Large doc (100 pages, ~500 chunks)

**Test file: `tests/test_embeddings.py`:**
```python
import pytest
from rag.embeddings import get_embedding, get_embeddings_batch

def test_single_embedding():
    text = "This is a test sentence."
    embedding = get_embedding(text)
    assert len(embedding) == 768  # all-mpnet-base-v2 dimension
    assert all(isinstance(x, float) for x in embedding)

def test_batch_embeddings():
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = get_embeddings_batch(texts)
    assert len(embeddings) == 3
    assert all(len(e) == 768 for e in embeddings)
```

---

### 5.2 Integration Tests

**End-to-end flows:**
1. Upload document → verify chunks in vector store
2. Query → verify correct chunks retrieved
3. Chat → verify LLM receives context
4. Delete document → verify cleanup

**Test file: `tests/test_integration.py`:**
```python
import pytest
from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

def test_document_lifecycle():
    # Upload
    with open("test_doc.txt", "rb") as f:
        response = client.post("/api/ingest", files={"file": f})
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]
    
    # List
    response = client.get("/api/documents")
    assert any(d["doc_id"] == doc_id for d in response.json())
    
    # Chat
    response = client.post("/api/chat", data={
        "message": "What is this document about?",
        "mode": "rag",
        "doc_ids": json.dumps([doc_id])
    })
    assert response.status_code == 200
    assert "content" in response.json()
    
    # Delete
    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 200
```

**Performance benchmarks:**
- Ingestion speed (chunks/second)
- Query latency (ms)
- Memory usage (MB)
- Storage efficiency (MB per document)

---

### 5.3 User Acceptance Testing

**Test scenarios:**
1. Upload Russia analysis doc → ask about sanctions
2. Upload multiple docs → ask cross-document question
3. Upload large doc (50 pages) → verify no memory issues
4. Delete doc → verify no orphaned data
5. Restart server → verify persistence

---

## Dependencies to Add

```txt
# requirements.txt additions
sentence-transformers>=2.2.0
chromadb>=0.4.0
tiktoken>=0.5.0          # for token counting
pypdf>=3.0.0             # for PDF support
torch>=2.0.0             # for MPS support
psutil>=5.9.0            # for memory monitoring
```

---

## Migration Strategy

### Option A: Parallel Implementation
- Keep current "direct mode" working
- Add RAG as new mode
- Users choose per-request

### Option B: Hybrid Auto-Detect
- Small files (<5k tokens) → direct mode
- Large files (>5k tokens) → auto-ingest to RAG
- Transparent to user

### Option C: Full Migration
- Replace file upload with document library
- All documents go through RAG
- Simpler UX, more consistent

**Recommendation:** Option A (parallel) for flexibility

---

## Rollout Plan

### Week 1: Core Infrastructure
- Implement embedding layer (`rag/embeddings.py`)
- Implement vector store wrapper (`rag/vector_store.py`)
- Implement chunking (`rag/chunking.py`)
- Unit tests

### Week 2: Backend API
- Extend FastAPI with new endpoints
- Integrate with existing chat endpoint
- Add document manager (`rag/document_manager.py`)
- Integration tests

### Week 3: Frontend
- Add document library UI
- Add RAG mode toggle
- Add source citations
- User testing

### Week 4: Polish & Deploy
- Performance optimization
- Error handling
- Documentation
- Deploy to M2 machine

---

## Success Metrics

**Performance:**
- Ingestion: >500 chunks/second
- Query: <200ms end-to-end
- Memory: <8GB total usage

**Quality:**
- Retrieval accuracy: >80% relevant chunks in top-5
- User satisfaction: Can answer questions from large docs
- No data leakage: All processing local

**Scalability:**
- Support 1000+ documents
- Support 50k+ chunks
- Handle 50MB files without issues

---

## Risk Mitigation

**Risk 1: Memory overflow on M2**
- Mitigation: Batch size limits, memory monitoring, cleanup

**Risk 2: Slow embedding on large docs**
- Mitigation: Background processing, progress indicators, async

**Risk 3: Poor retrieval quality**
- Mitigation: Tunable parameters, reranking, hybrid search

**Risk 4: Storage growth**
- Mitigation: Size limits, cleanup tools, compression

---

## Future Enhancements

1. **Multi-modal:** Support images, tables from PDFs
2. **Collaborative:** Share document library across team
3. **Advanced RAG:** Graph RAG, hierarchical retrieval
4. **Analytics:** Track which docs are most useful
5. **Export:** Export chat history with citations

---

## Reusable Patterns from Memory System

### From `Memory/knowledge-system/distillation/utils/embedding.py`:
- Batch processing pattern
- Client singleton pattern
- Token counting approach

### From `Memory/knowledge-system/distillation/storage/vector_client.py`:
- VectorClient interface design
- Metadata schema structure
- Batch upsert pattern
- Query with filtering

### Key Adaptations:
- OpenAI embeddings → Sentence Transformers
- Upstash Vector → Chroma
- Cloud storage → Local disk
- 3072 dimensions → 768 dimensions
- Batch size 100 → 32 (for M2)

---

This plan gives you a production-ready local RAG system that maintains your security requirements while scaling to handle large document collections on the M2 24GB machine.
