from pathlib import Path

from rag.document_manager import DocumentManager


def _make_manager(tmp_path: Path) -> DocumentManager:
    return DocumentManager(str(tmp_path / "uploaded_docs"))


def test_save_document_returns_doc_id(tmp_path: Path):
    manager = _make_manager(tmp_path)
    doc_id = manager.save_document(b"hello world", "sample.txt", "V1")
    assert doc_id.startswith("doc_")


def test_get_document_returns_metadata(tmp_path: Path):
    manager = _make_manager(tmp_path)
    doc_id = manager.save_document(b"hello world", "sample.txt", "V1", source_path="Research/sample.txt")
    doc = manager.get_document(doc_id)
    assert doc is not None
    assert doc["doc_id"] == doc_id
    assert doc["filename"] == "sample.txt"
    assert doc["source_path"] == "Research/sample.txt"
    assert doc["folder_path"] == "Research"
    assert doc["vertical"] == "V1"


def test_list_documents_filters_by_vertical(tmp_path: Path):
    manager = _make_manager(tmp_path)
    manager.save_document(b"doc one", "one.txt", "V1")
    manager.save_document(b"doc two", "two.txt", "V2")

    v1_docs = manager.list_documents("V1")
    v2_docs = manager.list_documents("V2")

    assert len(v1_docs) == 1
    assert v1_docs[0]["vertical"] == "V1"
    assert len(v2_docs) == 1
    assert v2_docs[0]["vertical"] == "V2"


def test_delete_document_removes_file_and_metadata(tmp_path: Path):
    manager = _make_manager(tmp_path)
    doc_id = manager.save_document(b"to delete", "delete.txt", "V1")
    doc = manager.get_document(doc_id)
    stored_path = Path(doc["stored_path"])
    assert stored_path.exists()

    manager.delete_document(doc_id)

    assert manager.get_document(doc_id) is None
    assert not stored_path.exists()


def test_is_unchanged_true_for_same_content_false_for_different(tmp_path: Path):
    manager = _make_manager(tmp_path)
    content = b"constant content"
    manager.save_document(content, "same.txt", "V1", source_path="Folder/same.txt")
    assert manager.is_unchanged(content, "same.txt", source_path="Folder/same.txt")
    assert not manager.is_unchanged(content, "same.txt", source_path="Other/same.txt")
    assert not manager.is_unchanged(b"updated content", "same.txt", source_path="Folder/same.txt")


def test_resaving_same_file_returns_same_doc_id(tmp_path: Path):
    manager = _make_manager(tmp_path)
    content = b"dedupe me"
    doc_id_1 = manager.save_document(content, "dedupe.txt", "V1")
    doc_id_2 = manager.save_document(content, "dedupe.txt", "V1")
    assert doc_id_1 == doc_id_2
    assert len(manager.list_documents()) == 1


def test_original_file_is_preserved_on_disk(tmp_path: Path):
    manager = _make_manager(tmp_path)
    content = b"preserve this file"
    doc_id = manager.save_document(content, "preserve.txt", "V1")
    doc = manager.get_document(doc_id)
    stored_path = Path(doc["stored_path"])
    assert stored_path.exists()
    assert stored_path.read_bytes() == content
