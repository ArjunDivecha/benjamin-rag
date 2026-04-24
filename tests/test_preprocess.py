from pathlib import Path

from docx import Document

import preprocess
from rag.document_manager import DocumentManager
from rag.vector_store import VectorStore


def _long_text(tokens: int) -> str:
    return " a" * tokens


def _write_docx(path: Path, text: str) -> None:
    doc = Document()
    doc.add_paragraph(text)
    doc.save(str(path))


def _base_args(storage: Path, vector: Path) -> list[str]:
    return ["--storage-path", str(storage), "--vector-path", str(vector)]


def _patch_embeddings(monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "get_embeddings_batch",
        lambda chunks: [[float(i), 0.0, 1.0] for i, _ in enumerate(chunks)],
    )


def test_ingest_txt_and_list_vertical(tmp_path: Path, capsys, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    txt = tmp_path / "brief.txt"
    txt.write_text(_long_text(130))

    rc = preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    assert rc == 0

    rc = preprocess.main(["--list", "V1", *_base_args(storage, vector)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "brief.txt" in out
    assert "vertical=V1" in out


def test_ingest_docx_chunk_count_positive(tmp_path: Path, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    docx_path = tmp_path / "guide.docx"
    _write_docx(docx_path, _long_text(180))

    rc = preprocess.main(["--vertical", "V2", "--files", str(docx_path), *_base_args(storage, vector)])
    assert rc == 0

    dm = DocumentManager(str(storage))
    docs = dm.list_documents("V2")
    assert len(docs) == 1
    assert docs[0]["chunk_count"] > 0


def test_reingest_same_file_skips_unchanged(tmp_path: Path, capsys, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    txt = tmp_path / "same.txt"
    txt.write_text(_long_text(140))

    preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    out = capsys.readouterr().out
    assert "skipped (unchanged)" in out


def test_modify_and_reingest_updates_chunks(tmp_path: Path, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    txt = tmp_path / "update.txt"

    txt.write_text(_long_text(120))
    preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    dm = DocumentManager(str(storage))
    first = dm.list_documents("V1")[0]
    first_chunks = first["chunk_count"]

    txt.write_text(_long_text(1200))
    preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    dm2 = DocumentManager(str(storage))
    docs = dm2.list_documents("V1")
    assert len(docs) == 1
    assert docs[0]["chunk_count"] > first_chunks


def test_remove_doc_id_removes_from_metadata_and_vector_store(tmp_path: Path, capsys, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    txt = tmp_path / "remove.txt"
    txt.write_text(_long_text(200))

    preprocess.main(["--vertical", "V1", "--files", str(txt), *_base_args(storage, vector)])
    dm = DocumentManager(str(storage))
    doc_id = dm.list_documents("V1")[0]["doc_id"]

    rc = preprocess.main(["--remove", doc_id, *_base_args(storage, vector)])
    assert rc == 0

    dm2 = DocumentManager(str(storage))
    assert dm2.get_document(doc_id) is None

    vs = VectorStore(str(vector))
    assert vs.get_collection("V1").count() == 0
    assert "removed" in capsys.readouterr().out


def test_dir_ingests_all_supported_files(tmp_path: Path, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    src = tmp_path / "src"
    src.mkdir()

    (src / "a.txt").write_text(_long_text(120))
    (src / "b.md").write_text(_long_text(130))
    _write_docx(src / "c.docx", _long_text(140))

    rc = preprocess.main(["--vertical", "V1", "--dir", str(src), *_base_args(storage, vector)])
    assert rc == 0

    dm = DocumentManager(str(storage))
    assert len(dm.list_documents("V1")) == 3


def test_dir_ingests_nested_files_with_relative_source_paths(tmp_path: Path, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    src = tmp_path / "src"
    nested = src / "Research" / "Calls"
    nested.mkdir(parents=True)

    (src / "root.txt").write_text(_long_text(120))
    (nested / "note.txt").write_text(_long_text(130))

    rc = preprocess.main(["--vertical", "V1", "--dir", str(src), *_base_args(storage, vector)])
    assert rc == 0

    dm = DocumentManager(str(storage))
    source_paths = {doc["source_path"] for doc in dm.list_documents("V1")}
    assert source_paths == {"root.txt", "Research/Calls/note.txt"}


def test_stats_shows_correct_counts_per_vertical(tmp_path: Path, capsys, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    f1 = tmp_path / "v1.txt"
    f2 = tmp_path / "v2.txt"
    f1.write_text(_long_text(120))
    f2.write_text(_long_text(140))

    preprocess.main(["--vertical", "V1", "--files", str(f1), *_base_args(storage, vector)])
    preprocess.main(["--vertical", "V2", "--files", str(f2), *_base_args(storage, vector)])

    rc = preprocess.main(["--stats", *_base_args(storage, vector)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "V1: docs=1" in out
    assert "V2: docs=1" in out
    assert "TOTAL: docs=2" in out


def test_unsupported_file_type_is_skipped_with_warning(tmp_path: Path, capsys, monkeypatch):
    _patch_embeddings(monkeypatch)
    storage = tmp_path / "uploaded_docs"
    vector = tmp_path / "chroma_db"
    txt = tmp_path / "ok.txt"
    bad = tmp_path / "bad.bin"
    txt.write_text(_long_text(120))
    bad.write_bytes(b"\x00\x01\x02")

    rc = preprocess.main(
        ["--vertical", "V1", "--files", str(txt), str(bad), *_base_args(storage, vector)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "bad.bin: skipped (unsupported file type)" in out

    dm = DocumentManager(str(storage))
    docs = dm.list_documents("V1")
    assert len(docs) == 1
    assert docs[0]["filename"] == "ok.txt"
