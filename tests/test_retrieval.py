from pathlib import Path

from rag.retrieval import assemble_context, retrieve_context
from rag.vector_store import VectorStore


def _seed_collection(store: VectorStore, collection_name: str, doc_id: str, filename: str, texts: list[str]):
    chunks = texts
    # Small synthetic embedding space for deterministic query behavior in tests.
    embeddings = []
    for t in texts:
        t_lower = t.lower()
        if "semiconductor" in t_lower:
            embeddings.append([1.0, 0.0, 0.0])
        elif "healthcare" in t_lower:
            embeddings.append([0.0, 1.0, 0.0])
        else:
            embeddings.append([0.0, 0.0, 1.0])
    store.upsert_document(
        collection_name=collection_name,
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        metadata={"filename": filename},
    )


def test_retrieve_relevant_chunks(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("V1")
    _seed_collection(
        store,
        "V1",
        "doc_1",
        "brief.txt",
        ["semiconductor demand is cyclical", "healthcare utilization rose"],
    )

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context("chips outlook", store, "V1", top_k=5, min_score=0.0)
    assert len(results) >= 1
    assert "semiconductor" in results[0]["text"].lower()


def test_min_score_filters_low_relevance(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("V1")
    _seed_collection(store, "V1", "doc_1", "brief.txt", ["general market commentary"])

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context("semiconductor", store, "V1", top_k=5, min_score=0.95)
    assert results == []


def test_top_k_limiting(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("V1")
    _seed_collection(
        store,
        "V1",
        "doc_1",
        "brief.txt",
        ["semiconductor a", "semiconductor b", "semiconductor c"],
    )
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context("semiconductor", store, "V1", top_k=2, min_score=0.0)
    assert len(results) <= 2


def test_assemble_context_xml_format():
    chunks = [
        {
            "text": "Example chunk text",
            "metadata": {"filename": "doc.txt", "chunk_id": 1, "total_chunks": 3},
        }
    ]
    context = assemble_context(chunks)
    assert '<document name="doc.txt" chunk="1/3">' in context
    assert "</document>" in context
    assert "Example chunk text" in context


def test_empty_collection_returns_empty(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("V1")
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context("anything", store, "V1", top_k=5, min_score=0.0)
    assert results == []


def test_collection_filtering_v1_vs_v2(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("V1")
    store.create_collection("V2")
    _seed_collection(store, "V1", "doc_1", "v1.txt", ["semiconductor insights"])
    _seed_collection(store, "V2", "doc_2", "v2.txt", ["healthcare insights"])

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    v1_results = retrieve_context("semiconductor", store, "V1", top_k=5, min_score=0.0)
    v2_results = retrieve_context("semiconductor", store, "V2", top_k=5, min_score=0.0)

    assert any("semiconductor" in r["text"].lower() for r in v1_results)
    assert all("semiconductor" not in r["text"].lower() for r in v2_results)
