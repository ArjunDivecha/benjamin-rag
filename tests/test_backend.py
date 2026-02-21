from pathlib import Path
import json
import io

from fastapi.testclient import TestClient

import backend

class _DummyResponse:
    def __init__(self, status_code: int, data: dict, text: str = ""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        return self._data

def _configure_rag_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(backend, "VECTOR_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(backend, "UPLOADED_DOCS_PATH", str(tmp_path / "uploaded_docs"))
    backend._reset_rag_services_for_tests()

def _seed_v1_doc(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    vs, dm = backend._ensure_rag_services()
    content = b" a" * 200
    doc_id = dm.save_document(content, "seed.txt", "V1", chunk_count=1)
    vs.upsert_document(
        collection_name="V1",
        doc_id=doc_id,
        chunks=["semiconductor screening examples"],
        embeddings=[[1.0, 0.0, 0.0]],
        metadata={"filename": "seed.txt", "vertical": "V1"},
    )
    return doc_id

def _seed_v3_doc(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    vs, dm = backend._ensure_rag_services()
    content = b" a" * 200
    doc_id = dm.save_document(content, "interview_notes.txt", "V3", chunk_count=1)
    vs.upsert_document(
        collection_name="V3",
        doc_id=doc_id,
        chunks=["Interviewee A said the market is growing fast despite volatility."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadata={"filename": "interview_notes.txt", "vertical": "V3"},
    )
    return doc_id

def test_chat_direct_mode_with_file_still_works(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Summarize this request briefly in 1 sentence. Make it super short.", "mode": "direct"},
        files={"file": ("brief.txt", "hello context", "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["content"], str)

def test_chat_rag_mode_returns_response_with_sources(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Draft a tiny 1-sentence brief exactly.",
            "mode": "rag",
            "objective": "expert_network_brief",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["content"], str)
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) >= 1

def test_chat_rag_empty_vertical_returns_error(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Any context?",
            "mode": "rag",
            "objective": "interview_guide",
        },
    )
    assert resp.status_code == 400

def test_chat_rag_insights_qa_uses_local_provider(tmp_path: Path, monkeypatch):
    _seed_v3_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])

    def fake_post(url, *args, **kwargs):
        return _DummyResponse(
            200,
            {
                "message": {"content": "insights ok"},
                "prompt_eval_count": 17,
                "eval_count": 9,
            },
        )
    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Who mentioned market growth?",
            "mode": "rag",
            "objective": "insights_qa",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["provider"] == "ollama"

def test_documents_endpoint_returns_list(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/documents")
    assert resp.status_code == 200, resp.text

def test_document_file_endpoint_returns_file(tmp_path: Path, monkeypatch):
    doc_id = _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get(f"/api/documents/{doc_id}/file")
    assert resp.status_code == 200, resp.text

def test_document_file_endpoint_missing_doc_returns_404(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/documents/doc_missing/file")
    assert resp.status_code == 404

def test_stats_endpoint_returns_expected_keys(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/stats")
    assert resp.status_code == 200, resp.text

def test_system_prompt_included_in_rag_payload(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Write a 5 word screening question.",
            "mode": "rag",
            "objective": "expert_network_brief",
        },
    )
    assert resp.status_code == 200, resp.text
    assert "content" in resp.json()

def test_provider_config_bedrock_direct(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    
    client = TestClient(backend.app)
    resp = client.post("/api/chat", data={"message": "Reply 'hello' directly.", "mode": "direct"})
    
    assert resp.status_code == 200, resp.text
    assert "hello" in resp.json()["content"].lower()

def test_local_model_config_reads_dotenv_when_env_missing(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOCAL_RAG_MODEL=qwen3:32b\n"
        "OLLAMA_BASE_URL=http://127.0.0.1:11434/\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LOCAL_RAG_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(backend, "APP_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    base_url, model = backend._load_local_model_config()
    assert base_url == "http://127.0.0.1:11434"
    assert model == "qwen3:32b"

def test_chat_compare_returns_local_and_opus(tmp_path: Path, monkeypatch):
    _seed_v3_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])

    def fake_post(url, *args, **kwargs):
        return _DummyResponse(
            200,
            {
                "message": {"content": "local ok"},
                "prompt_eval_count": 17,
                "eval_count": 9,
            },
        )
    monkeypatch.setattr(backend.requests, "post", fake_post)

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat/compare",
        data={"message": "In 1 sentence, who mentioned growth?", "objective": "insights_qa"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["local"]["content"] == "local ok"
    assert data["local"]["provider"] == "ollama"
    assert "content" in data["opus"]
    assert data["opus"]["provider"] == "bedrock"
