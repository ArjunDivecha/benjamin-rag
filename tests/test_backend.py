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


def test_documents_endpoint_returns_list(tmp_path: Path, monkeypatch):
    _seed_v1_doc(tmp_path, monkeypatch)
    client = TestClient(backend.app)
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["vertical"] == "V1"


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
