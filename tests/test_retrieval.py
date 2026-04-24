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


def test_filename_fallback_returns_chunk_when_semantic_search_misses(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("ALL")
    _seed_collection(
        store,
        "ALL",
        "doc_1",
        "Executive Summary.docx",
        ["global tariffs and trade policy analysis"],
    )
    _seed_collection(
        store,
        "ALL",
        "doc_2",
        "Interview Notes.docx",
        ["semiconductor utilization remains stable"],
    )

    # Simulate a query embedding that would not clear the semantic min_score threshold.
    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [0.0, 0.0, 1.0])
    results = retrieve_context("summarize this executive summary", store, "ALL", top_k=5, min_score=0.95)

    assert results
    assert results[0]["metadata"]["filename"] == "Executive Summary.docx"


def test_filename_match_beats_weak_semantic_hit(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("ALL")
    store.upsert_document(
        "ALL",
        "doc_exec",
        ["global tariffs and trade policy analysis"],
        [[0.0, 0.0, 1.0]],
        metadata={"filename": "Executive Summary.docx"},
    )
    store.upsert_document(
        "ALL",
        "doc_other",
        ["phase one wrap-up and software diligence findings"],
        [[0.6, 0.8, 0.0]],
        metadata={"filename": "5.1 Phase 1 Wrap-up.pdf"},
    )

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context("summarize this executive summary", store, "ALL", top_k=5, min_score=0.5)

    assert results
    assert all(result["metadata"]["filename"] == "Executive Summary.docx" for result in results)


def test_doc_id_scope_filters_retrieval(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("ALL")
    store.upsert_document(
        "ALL",
        "doc_alpha",
        ["alpha customer renewal notes"],
        [[1.0, 0.0, 0.0]],
        metadata={"filename": "alpha.txt", "source_path": "Alpha/alpha.txt"},
    )
    store.upsert_document(
        "ALL",
        "doc_beta",
        ["beta pricing pressure notes"],
        [[1.0, 0.0, 0.0]],
        metadata={"filename": "beta.txt", "source_path": "Beta/beta.txt"},
    )

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context(
        "pricing pressure",
        store,
        "ALL",
        top_k=5,
        min_score=0.0,
        doc_ids=["doc_beta"],
    )

    assert results
    assert {result["metadata"]["doc_id"] for result in results} == {"doc_beta"}


def test_folder_query_prefers_source_path_over_semantic_hit(tmp_path: Path, monkeypatch):
    store = VectorStore(str(tmp_path / "chroma"))
    store.create_collection("ALL")
    store.upsert_document(
        "ALL",
        "doc_two_a",
        ["expert brief ai services details"],
        [[0.0, 0.0, 1.0]],
        metadata={"filename": "2. Expert Network Brief - AI Services.docx", "source_path": "Two/2. Expert Network Brief - AI Services.docx", "folder_path": "Two"},
    )
    store.upsert_document(
        "ALL",
        "doc_two_b",
        ["expert brief nitrogen services details"],
        [[0.0, 0.0, 1.0]],
        metadata={"filename": "2. Expert Network Briefs - Nitrogen Services.docx", "source_path": "Two/2. Expert Network Briefs - Nitrogen Services.docx", "folder_path": "Two"},
    )
    store.upsert_document(
        "ALL",
        "doc_four",
        ["interview notes with a strong semantic hit"],
        [[1.0, 0.0, 0.0]],
        metadata={"filename": "4. Interview Notes.docx", "source_path": "Four/4. Interview Notes.docx", "folder_path": "Four"},
    )

    monkeypatch.setattr("rag.retrieval.get_embedding", lambda _: [1.0, 0.0, 0.0])
    results = retrieve_context(
        "give me details of the files in folder two",
        store,
        "ALL",
        top_k=5,
        min_score=0.0,
        doc_ids=["doc_two_a", "doc_two_b", "doc_four"],
    )

    assert results
    assert {result["metadata"]["folder_path"] for result in results} == {"Two"}
