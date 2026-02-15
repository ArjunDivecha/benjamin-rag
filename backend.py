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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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
OPUS_COMPARE_MODEL = "anthropic/claude-opus-4.6"
OLLAMA_BASE_URL = "http://localhost:11434"
LOCAL_RAG_MODEL = "qwen2.5:32b"
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
TEXT_ENCODINGS = ["utf-8", "latin-1", "cp1252"]
VECTOR_DB_PATH = os.path.join(APP_DIR, "chroma_db")
UPLOADED_DOCS_PATH = os.path.join(APP_DIR, "uploaded_docs")
OBJECTIVE_TO_COLLECTION = {
    "expert_network_brief": "V1",
    "interview_guide": "V2",
    "insights_qa": "V3",
}
OBJECTIVE_TO_PROVIDER = {
    "expert_network_brief": "openrouter",
    "interview_guide": "openrouter",
    "insights_qa": "ollama",
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
def _load_env_value_from_file(path: Path, key: str) -> Optional[str]:
    """Extract one key from a dotenv-style file."""
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith(f"{key}="):
            continue
        value = line.split("=", 1)[1].strip().strip('"\'')
        if value:
            return value
    return None


def _load_env_value(key: str) -> Optional[str]:
    """Load a value from environment, then local dotenv files."""
    value = os.environ.get(key, "").strip()
    if value:
        return value

    env_candidates = [
        Path(APP_DIR) / ".env",
        Path.cwd() / ".env",
        Path(APP_DIR) / ".env.txt",
        Path.cwd() / ".env.txt",
    ]
    for env_file in env_candidates:
        value = _load_env_value_from_file(env_file, key)
        if value:
            return value
    return None


def _load_openrouter_key() -> str:
    """Load OpenRouter API key from environment or local dotenv files."""
    key = _load_env_value("OPENROUTER_API_KEY")
    if key:
        return key

    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Create a local .env file with "
        "OPENROUTER_API_KEY=your_key"
    )


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


def _openrouter_chat(api_key: str, messages: list[dict], model: str = MODEL) -> dict:
    payload = {
        "model": model,
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


def _load_local_model_config() -> tuple[str, str]:
    base_url = (_load_env_value("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).strip().rstrip("/")
    model = (_load_env_value("LOCAL_RAG_MODEL") or LOCAL_RAG_MODEL).strip()
    if not base_url:
        base_url = OLLAMA_BASE_URL
    if not model:
        model = LOCAL_RAG_MODEL
    return base_url, model


def _load_opus_compare_model() -> str:
    model = (_load_env_value("OPUS_COMPARE_MODEL") or OPUS_COMPARE_MODEL).strip()
    return model or OPUS_COMPARE_MODEL


def _ollama_chat(messages: list[dict]) -> dict:
    base_url, model = _load_local_model_config()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=240)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Local LLM request failed: {exc}",
        ) from exc

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error") or err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise HTTPException(status_code=resp.status_code, detail=msg)

    data = resp.json()
    message = data.get("message", {}) if isinstance(data, dict) else {}
    content = str(message.get("content", "")).strip()
    return {
        "content": content,
        "usage": {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        },
        "timing": {
            "total_duration_s": float(data.get("total_duration") or 0) / 1_000_000_000.0,
            "load_duration_s": float(data.get("load_duration") or 0) / 1_000_000_000.0,
            "prompt_eval_duration_s": float(data.get("prompt_eval_duration") or 0) / 1_000_000_000.0,
            "eval_duration_s": float(data.get("eval_duration") or 0) / 1_000_000_000.0,
        },
        "model": model,
    }


def _build_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    eval_duration_s: Optional[float] = None,
) -> dict:
    total_tokens = int(prompt_tokens) + int(completion_tokens)
    tok_per_sec = 0.0
    if eval_duration_s and eval_duration_s > 0 and completion_tokens > 0:
        tok_per_sec = float(completion_tokens) / float(eval_duration_s)
    elif latency_s > 0 and completion_tokens > 0:
        tok_per_sec = float(completion_tokens) / float(latency_s)

    return {
        "total_tokens": total_tokens,
        "latency_ms": round(float(latency_s) * 1000.0, 1),
        "tok_per_sec": round(tok_per_sec, 2) if tok_per_sec > 0 else 0.0,
    }


def _build_rag_summary(collection_name: str, chunks: list[dict]) -> dict:
    docs_retrieved = {
        c.get("metadata", {}).get("doc_id")
        for c in chunks
        if c.get("metadata", {}).get("doc_id")
    }
    scores = [float(c.get("score", 0.0)) for c in chunks]
    return {
        "collection": collection_name,
        "chunks_retrieved": len(chunks),
        "documents_retrieved": len(docs_retrieved),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
    }


def _build_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": c.get("metadata", {}).get("doc_id"),
            "chunk_id": c.get("metadata", {}).get("chunk_id"),
            "score": c.get("score", 0.0),
            "filename": c.get("metadata", {}).get("filename"),
            "snippet": (c.get("text", "") or "").strip()[:280],
        }
        for c in chunks
    ]


def _run_openrouter_model(messages: list[dict], model_name: str, api_key: Optional[str] = None) -> dict:
    if not api_key:
        try:
            api_key = _load_openrouter_key()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    llm_started = time.perf_counter()
    data = _openrouter_chat(
        api_key=api_key,
        messages=messages,
        model=model_name,
    )
    llm_elapsed_s = time.perf_counter() - llm_started
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "content": content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "provider": "openrouter",
        "model": model_name,
        "metrics": _build_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=llm_elapsed_s,
        ),
    }


def _run_ollama_model(messages: list[dict]) -> dict:
    llm_started = time.perf_counter()
    local = _ollama_chat(messages=messages)
    llm_elapsed_s = time.perf_counter() - llm_started
    content = local.get("content", "")
    usage = local.get("usage", {})
    eval_duration_s = float(local.get("timing", {}).get("eval_duration_s") or 0.0)
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "content": content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "provider": "ollama",
        "model": local.get("model", LOCAL_RAG_MODEL),
        "metrics": _build_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=llm_elapsed_s,
            eval_duration_s=eval_duration_s,
        ),
    }


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
    request_started = time.perf_counter()
    if mode not in {"direct", "rag"}:
        raise HTTPException(status_code=400, detail="mode must be 'direct' or 'rag'")

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
        response = _run_openrouter_model(
            messages=[{"role": "user", "content": user_content}],
            model_name=MODEL,
        )
        return response

    if not objective or objective not in OBJECTIVE_TO_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail="objective must be 'expert_network_brief', 'interview_guide', or 'insights_qa' in rag mode",
        )

    try:
        vector_store, _ = _ensure_rag_services()
        from rag.retrieval import assemble_context, retrieve_context
        from rag.system_prompts import load_system_prompt
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    collection_name = OBJECTIVE_TO_COLLECTION[objective]
    retrieval_started = time.perf_counter()
    chunks = retrieve_context(
        query=message.strip(),
        vector_store=vector_store,
        collection_name=collection_name,
        top_k=max(1, min(top_k, 20)),
        min_score=min_score,
    )
    retrieval_elapsed_s = time.perf_counter() - retrieval_started
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

    provider = OBJECTIVE_TO_PROVIDER.get(objective, "openrouter")
    model_response = None
    if provider == "openrouter":
        model_response = _run_openrouter_model(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model_name=MODEL,
        )
    elif provider == "ollama":
        model_response = _run_ollama_model(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported provider: {provider}")

    rag_summary = _build_rag_summary(collection_name=collection_name, chunks=chunks)
    sources = _build_sources(chunks)
    request_elapsed_s = time.perf_counter() - request_started
    model_response["metrics"]["retrieval_ms"] = round(retrieval_elapsed_s * 1000.0, 1)
    model_response["metrics"]["request_total_ms"] = round(request_elapsed_s * 1000.0, 1)

    return {
        **model_response,
        "rag": rag_summary,
        "sources": sources,
    }


@app.post("/api/chat/compare")
async def chat_compare(
    message: str = Form(..., description="User message/prompt"),
    objective: str = Form("insights_qa"),
    top_k: int = Form(5),
    min_score: float = Form(0.5),
):
    """
    Compare endpoint for side-by-side RAG answers:
    - local ollama model
    - OpenRouter Opus model
    """
    request_started = time.perf_counter()
    objective = (objective or "").strip()
    if objective != "insights_qa":
        raise HTTPException(
            status_code=400,
            detail="compare endpoint currently supports objective='insights_qa' only",
        )

    try:
        vector_store, _ = _ensure_rag_services()
        from rag.retrieval import assemble_context, retrieve_context
        from rag.system_prompts import load_system_prompt
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    collection_name = OBJECTIVE_TO_COLLECTION[objective]
    retrieval_started = time.perf_counter()
    chunks = retrieve_context(
        query=message.strip(),
        vector_store=vector_store,
        collection_name=collection_name,
        top_k=max(1, min(top_k, 20)),
        min_score=min_score,
    )
    retrieval_elapsed_s = time.perf_counter() - retrieval_started
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
    shared_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        api_key = _load_openrouter_key()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    opus_model = _load_opus_compare_model()

    with ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(_run_ollama_model, shared_messages)
        opus_future = executor.submit(
            _run_openrouter_model,
            shared_messages,
            opus_model,
            api_key,
        )
        local_result = local_future.result()
        opus_result = opus_future.result()

    rag_summary = _build_rag_summary(collection_name=collection_name, chunks=chunks)
    sources = _build_sources(chunks)
    retrieval_ms = round(retrieval_elapsed_s * 1000.0, 1)
    request_elapsed_s = time.perf_counter() - request_started
    local_result["metrics"]["retrieval_ms"] = retrieval_ms
    opus_result["metrics"]["retrieval_ms"] = retrieval_ms

    return {
        "local": local_result,
        "opus": opus_result,
        "rag": rag_summary,
        "sources": sources,
        "metrics": {
            "retrieval_ms": retrieval_ms,
            "request_total_ms": round(request_elapsed_s * 1000.0, 1),
        },
    }


@app.get("/api/documents")
def documents(vertical: Optional[str] = None):
    try:
        _, doc_manager = _ensure_rag_services()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return doc_manager.list_documents(vertical=vertical)


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: str):
    try:
        _, doc_manager = _ensure_rag_services()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    doc = doc_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    stored_path = Path(str(doc.get("stored_path", "")))
    if not stored_path.exists() or not stored_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Stored file missing for document '{doc_id}'",
        )

    return FileResponse(
        path=str(stored_path),
        filename=doc.get("filename") or stored_path.name,
    )


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
