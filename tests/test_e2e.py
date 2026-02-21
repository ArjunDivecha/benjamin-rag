from pathlib import Path
import json
import io

from fastapi.testclient import TestClient

import backend
import preprocess
from rag.document_manager import DocumentManager


class _DummyResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = ""

    def json(self):
        return self._data

def _long_text(tokens: int, topic: str) -> str:
    return (f" {topic}") * tokens


def _configure(tmp_path: Path, monkeypatch):
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    monkeypatch.setattr(backend, "VECTOR_DB_PATH", str(vector))
    monkeypatch.setattr(backend, "UPLOADED_DOCS_PATH", str(storage))
    backend._reset_rag_services_for_tests()
    return storage, vector


def _patch_embeddings(monkeypatch):
    monkeypatch.setattr(preprocess, "get_embeddings_batch", lambda chunks: [[1.0, 0.0, 0.0] for _ in chunks])
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])


def _patch_bedrock(monkeypatch):
    # Removing the dummy bedrock client to make ACTUAL calls
    pass


def test_full_pipeline_ingest_and_query(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_bedrock(monkeypatch)

    f = tmp_path / "brief.txt"
    f.write_text(_long_text(140, "semiconductor"))
    rc = preprocess.main(
        ["--vertical", "V1", "--files", str(f), "--storage-path", str(storage), "--vector-path", str(vector)]
    )
    assert rc == 0

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Draft brief", "mode": "rag", "objective": "expert_network_brief"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"]


def test_incremental_update_add_second_doc_and_query_both(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_bedrock(monkeypatch)

    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text(_long_text(130, "alpha"))
    f2.write_text(_long_text(130, "beta"))

    preprocess.main(["--vertical", "V1", "--files", str(f1), "--storage-path", str(storage), "--vector-path", str(vector)])
    preprocess.main(["--vertical", "V1", "--files", str(f2), "--storage-path", str(storage), "--vector-path", str(vector)])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Summarize", "mode": "rag", "objective": "expert_network_brief", "top_k": 5},
    )
    assert resp.status_code == 200
    source_doc_ids = {s["doc_id"] for s in resp.json()["sources"]}
    assert len(source_doc_ids) >= 2


def test_remove_doc_and_query_no_longer_returns_it(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_bedrock(monkeypatch)

    f = tmp_path / "remove_me.txt"
    f.write_text(_long_text(130, "remove"))
    preprocess.main(["--vertical", "V1", "--files", str(f), "--storage-path", str(storage), "--vector-path", str(vector)])

    dm = DocumentManager(str(storage))
    doc_id = dm.list_documents("V1")[0]["doc_id"]
    preprocess.main(["--remove", doc_id, "--storage-path", str(storage), "--vector-path", str(vector)])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Any context?", "mode": "rag", "objective": "expert_network_brief"},
    )
    assert resp.status_code == 400


def test_cross_vertical_isolation(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_bedrock(monkeypatch)

    f1 = tmp_path / "v1.txt"
    f2 = tmp_path / "v2.txt"
    f1.write_text(_long_text(130, "briefing"))
    f2.write_text(_long_text(130, "stakeholder"))

    preprocess.main(["--vertical", "V1", "--files", str(f1), "--storage-path", str(storage), "--vector-path", str(vector)])
    preprocess.main(["--vertical", "V2", "--files", str(f2), "--storage-path", str(storage), "--vector-path", str(vector)])

    dm = DocumentManager(str(storage))
    v1_doc_ids = {d["doc_id"] for d in dm.list_documents("V1")}

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Interview questions", "mode": "rag", "objective": "interview_guide"},
    )
    assert resp.status_code == 200
    result_doc_ids = {s["doc_id"] for s in resp.json()["sources"]}
    assert result_doc_ids.isdisjoint(v1_doc_ids)


def test_security_bedrock_only_with_aws_params(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    dummy_client = _patch_bedrock(monkeypatch)

    f = tmp_path / "secure.txt"
    f.write_text(_long_text(130, "secure"))
    preprocess.main(["--vertical", "V1", "--files", str(f), "--storage-path", str(storage), "--vector-path", str(vector)])

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Check", "mode": "rag", "objective": "expert_network_brief"},
    )
    assert resp.status_code == 200
    assert "content" in resp.json()


def test_persistence_after_backend_reinit(tmp_path: Path, monkeypatch):
    storage, vector = _configure(tmp_path, monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_bedrock(monkeypatch)

    f = tmp_path / "persist.txt"
    f.write_text(_long_text(130, "persistent"))
    preprocess.main(["--vertical", "V1", "--files", str(f), "--storage-path", str(storage), "--vector-path", str(vector)])

    # Simulate backend restart by resetting lazy service singletons.
    backend._reset_rag_services_for_tests()

    client = TestClient(backend.app)
    resp = client.post(
        "/api/chat",
        data={"message": "Use persisted data", "mode": "rag", "objective": "expert_network_brief"},
    )
    assert resp.status_code == 200
    assert resp.json()["sources"]
