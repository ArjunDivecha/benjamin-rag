from pathlib import Path

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
    captured = {}

    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _DummyResponse(
            200,
            {
                "choices": [{"message": {"content": "direct ok"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Summarize this", "mode": "direct"},
        files={"file": ("brief.txt", "hello context", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "direct ok"
    assert "<file name=\"brief.txt\">" in captured["payload"]["messages"][0]["content"]


def test_chat_rag_mode_returns_response_with_sources(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")

    def fake_post(url, headers, json, timeout):
        return _DummyResponse(
            200,
            {
                "choices": [{"message": {"content": "rag ok"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 12},
            },
        )

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Draft a brief",
            "mode": "rag",
            "objective": "expert_network_brief",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "rag ok"
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["doc_id"].startswith("doc_")


def test_chat_rag_empty_vertical_returns_error(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")
    monkeypatch.setattr(
        backend.requests,
        "post",
        lambda *args, **kwargs: _DummyResponse(200, {"choices": [{"message": {"content": "unused"}}]}),
    )

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
    assert "No relevant context" in resp.json()["detail"]


def test_chat_rag_insights_qa_uses_local_provider(tmp_path: Path, monkeypatch):
    _seed_v3_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    captured = {}

    def fake_post(url, *args, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs.get("json", {})
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
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "insights ok"
    assert data["provider"] == "ollama"
    assert data["usage"]["prompt_tokens"] == 17
    assert data["usage"]["completion_tokens"] == 9
    assert data["sources"]
    assert captured["url"].endswith("/api/chat")
    assert "messages" in captured["payload"]


def test_documents_endpoint_returns_list(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["vertical"] == "V1"


def test_document_file_endpoint_returns_file(tmp_path: Path, monkeypatch):
    doc_id = _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get(f"/api/documents/{doc_id}/file")
    assert resp.status_code == 200
    assert resp.content == (b" a" * 200)
    assert "seed.txt" in resp.headers.get("content-disposition", "")


def test_document_file_endpoint_missing_doc_returns_404(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/documents/doc_missing/file")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_stats_endpoint_returns_expected_keys(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "by_vertical" in data


def test_system_prompt_included_in_rag_payload(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        return _DummyResponse(
            200,
            {"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={
            "message": "Write screening questions",
            "mode": "rag",
            "objective": "expert_network_brief",
        },
    )
    assert resp.status_code == 200
    messages = captured["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert "screening" in messages[0]["content"].lower()


def test_provider_config_and_only_openrouter_call(tmp_path: Path, monkeypatch):
    _configure_rag_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "payload": json})
        return _DummyResponse(
            200,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post("/api/chat", data={"message": "hello", "mode": "direct"})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0]["url"] == backend.OPENROUTER_URL
    provider = calls[0]["payload"]["provider"]
    assert provider["zdr"] is True
    assert provider["order"] == ["amazon-bedrock"]
    assert provider["allow_fallbacks"] is False


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
    monkeypatch.setattr(backend, "_load_openrouter_key", lambda: "test-key")
    monkeypatch.setattr(backend, "_load_opus_compare_model", lambda: "anthropic/claude-opus-4.6")
    calls = []

    def fake_post(url, *args, **kwargs):
        calls.append(url)
        if url == backend.OPENROUTER_URL:
            return _DummyResponse(
                200,
                {
                    "choices": [{"message": {"content": "opus ok"}}],
                    "usage": {"prompt_tokens": 29, "completion_tokens": 11},
                },
            )
        if str(url).endswith("/api/chat"):
            return _DummyResponse(
                200,
                {
                    "message": {"content": "local ok"},
                    "prompt_eval_count": 17,
                    "eval_count": 9,
                },
            )
        return _DummyResponse(404, {}, text="unexpected")

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat/compare",
        data={"message": "Who mentioned growth?", "objective": "insights_qa"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["local"]["content"] == "local ok"
    assert data["local"]["provider"] == "ollama"
    assert data["opus"]["content"] == "opus ok"
    assert data["opus"]["provider"] == "openrouter"
    assert data["rag"]["collection"] == "V3"
    assert data["sources"]
    assert backend.OPENROUTER_URL in calls
