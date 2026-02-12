from pathlib import Path

from rag.vector_store import VectorStore


def _make_store(tmp_path: Path) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma_db"))


def _sample_doc():
    chunks = ["alpha", "beta", "gamma"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    metadata = {"filename": "test.txt", "vertical": "V1"}
    return chunks, embeddings, metadata


def test_create_collection_exists(tmp_path: Path):
    store = _make_store(tmp_path)
    col = store.create_collection("V1")
    assert col is not None
    assert store.get_collection("V1").name == col.name


def test_upsert_chunks_and_count(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)
    assert store.get_collection("V1").count() == 3


def test_search_returns_scores(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)

    results = store.search("V1", query_embedding=[1.0, 0.0, 0.0], top_k=2, min_score=0.0)
    assert len(results) > 0
    assert "score" in results[0]
    assert "text" in results[0]
    assert "metadata" in results[0]


def test_delete_document_removes_vectors(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)
    store.delete_document("V1", "test_1")
    assert store.get_collection("V1").count() == 0


def test_idempotent_upsert_same_doc_id(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)
    count_1 = store.get_collection("V1").count()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)
    count_2 = store.get_collection("V1").count()
    assert count_1 == count_2 == 3


def test_collections_are_isolated(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    store.create_collection("V2")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)

    assert store.get_collection("V1").count() == 3
    assert store.get_collection("V2").count() == 0


def test_get_stats_counts_vectors_and_docs(tmp_path: Path):
    store = _make_store(tmp_path)
    store.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store.upsert_document("V1", "test_1", chunks, embeddings, metadata)
    stats = store.get_stats("V1")
    assert stats["vector_count"] == 3
    assert stats["doc_count"] == 1


def test_persistence_after_reopen(tmp_path: Path):
    db_path = tmp_path / "chroma_db"
    store1 = VectorStore(str(db_path))
    store1.create_collection("V1")
    chunks, embeddings, metadata = _sample_doc()
    store1.upsert_document("V1", "test_1", chunks, embeddings, metadata)

    store2 = VectorStore(str(db_path))
    assert store2.get_collection("V1").count() == 3
