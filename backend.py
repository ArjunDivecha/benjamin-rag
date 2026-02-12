"""
=============================================================================
SCRIPT NAME: backend.py
=============================================================================

INPUT FILES: None (receives file uploads via API)

OUTPUT FILES: None (returns JSON responses)

VERSION: 1.0
LAST UPDATED: 2026-02-11

DESCRIPTION:
FastAPI backend that proxies requests to OpenRouter Claude Sonnet 4.5
(Bedrock + ZDR). Supports direct file-context chat and RAG chat mode.

DEPENDENCIES:
- fastapi, uvicorn, python-multipart, requests

USAGE:
uvicorn backend:app --reload --port 8000

Then open http://localhost:8000
=============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add python_utils for 1Password
sys.path.insert(0, "/Users/arjundivecha/python_utils")
try:
    from onepassword_credentials import load_credentials, OnePasswordError
except ImportError:
    load_credentials = None
    OnePasswordError = Exception

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.5"
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
TEXT_ENCODINGS = ["utf-8", "latin-1", "cp1252"]
VECTOR_DB_PATH = os.path.join(APP_DIR, "chroma_db")
UPLOADED_DOCS_PATH = os.path.join(APP_DIR, "uploaded_docs")
OBJECTIVE_TO_COLLECTION = {
    "expert_network_brief": "V1",
    "interview_guide": "V2",
}

_vector_store = None
_doc_manager = None
_rag_import_error = None

app = FastAPI(title="Sonnet + File Context", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Credentials
# -----------------------------------------------------------------------------
def _load_openrouter_key() -> str:
    """Load OpenRouter API key from 1Password or fallback."""
    if load_credentials:
        try:
            load_credentials(["OpenRouter"], verbose=False)
        except OnePasswordError:
            pass
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = "/Users/arjundivecha/Dropbox/AAA Backup/.env.txt"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError("OPENROUTER_API_KEY not found")


# -----------------------------------------------------------------------------
# File reading
# -----------------------------------------------------------------------------
def _read_docx(content: bytes) -> str:
    """Extract text from .docx file."""
    from io import BytesIO
    from docx import Document
    doc = Document(BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_file_content(content: bytes, filename: str) -> str:
    """Decode file bytes to string. Handles .docx and plain text."""
    lower = filename.lower()
    if lower.endswith(".docx"):
        return _read_docx(content)
    for enc in TEXT_ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filename} as text (tried {TEXT_ENCODINGS})")


# -----------------------------------------------------------------------------
# RAG helpers
# -----------------------------------------------------------------------------
def _ensure_rag_services():
    global _vector_store, _doc_manager, _rag_import_error

    if _vector_store is not None and _doc_manager is not None:
        return _vector_store, _doc_manager

    if _rag_import_error is not None:
        raise RuntimeError(str(_rag_import_error))

    try:
        from rag.vector_store import VectorStore
        from rag.document_manager import DocumentManager
    except Exception as exc:
        _rag_import_error = exc
        raise RuntimeError(f"RAG dependencies unavailable: {exc}") from exc

    Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
    Path(UPLOADED_DOCS_PATH).mkdir(parents=True, exist_ok=True)
    _vector_store = VectorStore(VECTOR_DB_PATH)
    _doc_manager = DocumentManager(UPLOADED_DOCS_PATH)
    return _vector_store, _doc_manager


def _reset_rag_services_for_tests() -> None:
    global _vector_store, _doc_manager, _rag_import_error
    _vector_store = None
    _doc_manager = None
    _rag_import_error = None


def _openrouter_chat(api_key: str, messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "provider": {
            "order": ["amazon-bedrock"],
            "zdr": True,
            "allow_fallbacks": False,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise HTTPException(status_code=resp.status_code, detail=msg)
    return resp.json()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(
    message: str = Form(..., description="User message/prompt"),
    file: Optional[UploadFile] = File(None),
    mode: str = Form("direct"),
    objective: Optional[str] = Form(None),
    top_k: int = Form(5),
    min_score: float = Form(0.5),
):
    """
    Chat endpoint with two modes:
    - direct: existing file-upload behavior
    - rag: retrieve from local vector store and include sources
    """
    mode = (mode or "direct").strip().lower()
    if mode not in {"direct", "rag"}:
        raise HTTPException(status_code=400, detail="mode must be 'direct' or 'rag'")

    try:
        api_key = _load_openrouter_key()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if mode == "direct":
        # Build user content: optional file context + message
        user_content_parts = []

        if file and file.filename:
            raw = await file.read()
            if len(raw) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large (max {MAX_FILE_SIZE // 1024}KB)",
                )
            try:
                text = _read_file_content(raw, file.filename)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            user_content_parts.append(f"<file name=\"{file.filename}\">\n{text}\n</file>")

        user_content_parts.append(message.strip())
        user_content = "\n\n" + "\n\n".join(user_content_parts)
        data = _openrouter_chat(
            api_key=api_key,
            messages=[{"role": "user", "content": user_content}],
        )
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }

    if not objective or objective not in OBJECTIVE_TO_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail="objective must be 'expert_network_brief' or 'interview_guide' in rag mode",
        )

    try:
        vector_store, _ = _ensure_rag_services()
        from rag.retrieval import assemble_context, retrieve_context
        from rag.system_prompts import load_system_prompt
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    collection_name = OBJECTIVE_TO_COLLECTION[objective]
    chunks = retrieve_context(
        query=message.strip(),
        vector_store=vector_store,
        collection_name=collection_name,
        top_k=max(1, min(top_k, 20)),
        min_score=min_score,
    )
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No relevant context found in collection '{collection_name}'",
        )

    context = assemble_context(chunks)
    system_prompt = load_system_prompt(objective)
    user_content = (
        "<retrieved_context>\n"
        f"{context}\n"
        "</retrieved_context>\n\n"
        f"User request:\n{message.strip()}"
    )

    data = _openrouter_chat(
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})

    return {
        "content": content,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "sources": [
            {
                "doc_id": c.get("metadata", {}).get("doc_id"),
                "chunk_id": c.get("metadata", {}).get("chunk_id"),
                "score": c.get("score", 0.0),
                "filename": c.get("metadata", {}).get("filename"),
            }
            for c in chunks
        ],
    }


@app.get("/api/documents")
def documents(vertical: Optional[str] = None):
    try:
        _, doc_manager = _ensure_rag_services()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return doc_manager.list_documents(vertical=vertical)


@app.get("/api/stats")
def stats():
    try:
        vector_store, doc_manager = _ensure_rag_services()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    docs = doc_manager.list_documents()
    total_chunks = sum(int(d["chunk_count"]) for d in docs)
    by_vertical = {}
    for d in docs:
        vertical = d["vertical"]
        entry = by_vertical.setdefault(vertical, {"doc_count": 0, "chunk_count": 0})
        entry["doc_count"] += 1
        entry["chunk_count"] += int(d["chunk_count"])

    for vertical in by_vertical:
        by_vertical[vertical]["vector_count"] = vector_store.get_stats(vertical)["vector_count"]

    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "by_vertical": by_vertical,
    }


@app.get("/")
def index():
    """Serve the frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Static files not found. Run from Junk directory."}


# Mount static files (for CSS, JS if any)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
