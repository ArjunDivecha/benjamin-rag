"""CLI for local RAG preprocessing: ingest, list, remove, and stats."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from rag.chunking import chunk_text
from rag.document_manager import DocumentManager
from rag.embeddings import get_embeddings_batch
from rag.vector_store import VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf", ".csv", ".json", ".xml", ".yaml", ".yml"}
TEXT_ENCODINGS = ["utf-8", "latin-1", "cp1252"]


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _read_text_bytes(raw: bytes, filename: str) -> str:
    for enc in TEXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filename} as text")


def read_file_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raw = path.read_bytes()
    return _read_text_bytes(raw, path.name)


def _resolve_files(files: List[str] | None, directory: str | None) -> List[Path]:
    resolved: List[Path] = []
    if files:
        resolved.extend(Path(f).expanduser().resolve() for f in files)
    if directory:
        dir_path = Path(directory).expanduser().resolve()
        for p in sorted(dir_path.rglob("*")):
            if p.is_file():
                resolved.append(p)
    deduped = []
    seen = set()
    for p in resolved:
        if str(p) in seen:
            continue
        seen.add(str(p))
        deduped.append(p)
    return deduped


def _source_path_for(path: Path, base_dir: str | Path | None = None) -> str:
    if base_dir:
        try:
            relative = path.resolve().relative_to(Path(base_dir).expanduser().resolve())
            return relative.as_posix()
        except ValueError:
            pass
    return path.name


def _folder_path_for(source_path: str) -> str:
    parent = Path(source_path).parent
    if str(parent) == ".":
        return ""
    return parent.as_posix()


def ingest_files(
    vertical: str,
    files: Iterable[Path],
    vector_store: VectorStore,
    doc_manager: DocumentManager,
    base_dir: str | Path | None = None,
) -> int:
    ingested = 0
    vector_store.create_collection(vertical)

    for path in files:
        filename = path.name
        source_path = _source_path_for(path, base_dir=base_dir)
        folder_path = _folder_path_for(source_path)
        suffix = path.suffix.lower()

        if not path.exists() or not path.is_file():
            print(f"{path}: skipped (not found)")
            continue
        if suffix not in SUPPORTED_SUFFIXES:
            print(f"{source_path}: skipped (unsupported file type)")
            continue

        started = time.time()
        file_bytes = path.read_bytes()

        if doc_manager.is_unchanged(
            file_bytes,
            filename,
            vertical=vertical,
            source_path=source_path,
        ):
            print(f"{source_path}: skipped (unchanged)")
            continue

        # Replace prior versions of same file in this vertical.
        for existing in doc_manager.list_documents(vertical=vertical):
            if (existing.get("source_path") or existing["filename"]) == source_path:
                vector_store.delete_document(vertical, existing["doc_id"])
                doc_manager.delete_document(existing["doc_id"])

        try:
            text = read_file_content(path)
        except Exception as exc:
            print(f"{source_path}: skipped (read error: {exc})")
            continue

        chunks = chunk_text(text)
        embeddings = get_embeddings_batch(chunks) if chunks else []

        doc_id = doc_manager.save_document(
            file_bytes=file_bytes,
            filename=filename,
            vertical=vertical,
            chunk_count=len(chunks),
            source_path=source_path,
        )
        vector_store.upsert_document(
            collection_name=vertical,
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata={
                "filename": filename,
                "source_path": source_path,
                "folder_path": folder_path,
                "vertical": vertical,
            },
        )

        elapsed = time.time() - started
        print(f"{source_path}: ingested ({len(chunks)} chunks, {elapsed:.2f}s, doc_id={doc_id})")
        ingested += 1

    return ingested


def sync_collection(
    vertical: str,
    files: Iterable[Path],
    vector_store: VectorStore,
    doc_manager: DocumentManager,
    base_dir: str | Path | None = None,
) -> tuple[int, int]:
    """Sync the RAG to exactly match the provided files.

    1. Ingest new/changed files.
    2. Remove any documents in this vertical whose source path is not in the file list.
    """
    file_list = list(files)
    ingested = ingest_files(vertical, file_list, vector_store, doc_manager, base_dir=base_dir)

    current_source_paths = set()
    for p in file_list:
        suffix = p.suffix.lower()
        if p.exists() and p.is_file() and suffix in SUPPORTED_SUFFIXES:
            current_source_paths.add(_source_path_for(p, base_dir=base_dir))

    removed = 0
    for doc in doc_manager.list_documents(vertical=vertical):
        doc_source_path = doc.get("source_path") or doc["filename"]
        if doc_source_path not in current_source_paths:
            vector_store.delete_document(vertical, doc["doc_id"])
            doc_manager.delete_document(doc["doc_id"])
            print(f"{doc_source_path}: removed (no longer in directory)")
            removed += 1

    if removed:
        print(f"Sync complete: {ingested} ingested, {removed} removed")
    return ingested, removed


def remove_document(doc_id: str, vector_store: VectorStore, doc_manager: DocumentManager) -> int:
    doc = doc_manager.get_document(doc_id)
    if not doc:
        print(f"{doc_id}: not found")
        return 1
    vertical = doc["vertical"]
    vector_store.delete_document(vertical, doc_id)
    doc_manager.delete_document(doc_id)
    print(f"{doc_id}: removed")
    return 0


def list_documents(doc_manager: DocumentManager, vertical: str | None = None) -> int:
    docs = doc_manager.list_documents(vertical=vertical)
    if not docs:
        print("No documents found.")
        return 0
    for d in docs:
        source_path = d.get("source_path") or d["filename"]
        print(
            f'{d["doc_id"]} | vertical={d["vertical"]} | path={source_path} | chunks={d["chunk_count"]}'
        )
    return 0


def print_stats(vector_store: VectorStore, doc_manager: DocumentManager) -> int:
    docs = doc_manager.list_documents()
    if not docs:
        print("No documents found.")
        return 0

    by_vertical = {}
    for d in docs:
        vertical = d["vertical"]
        entry = by_vertical.setdefault(vertical, {"doc_count": 0, "chunk_count": 0})
        entry["doc_count"] += 1
        entry["chunk_count"] += int(d["chunk_count"])

    for vertical, entry in sorted(by_vertical.items()):
        vector_count = vector_store.get_stats(vertical)["vector_count"]
        print(
            f"{vertical}: docs={entry['doc_count']} chunks={entry['chunk_count']} vectors={vector_count}"
        )

    print(f"TOTAL: docs={len(docs)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG preprocessing CLI")
    parser.add_argument("--vertical", type=str, help="Vertical/collection name (e.g. V1 or V2)")
    parser.add_argument("--files", nargs="+", help="List of files to ingest")
    parser.add_argument("--dir", dest="directory", help="Directory to ingest all supported files from")
    parser.add_argument("--remove", type=str, help="Remove a document by doc_id")
    parser.add_argument("--list", dest="list_vertical", nargs="?", const="__ALL__", help="List documents")
    parser.add_argument("--stats", action="store_true", help="Show per-vertical stats")
    parser.add_argument("--storage-path", default="uploaded_docs", help="Local document storage directory")
    parser.add_argument("--vector-path", default="chroma_db", help="Local Chroma persist directory")
    parser.add_argument("--sync", action="store_true",
                        help="Sync mode: ingest new/changed files AND remove documents no longer in the directory")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    vector_store = VectorStore(args.vector_path)
    doc_manager = DocumentManager(args.storage_path)

    if args.remove:
        return remove_document(args.remove, vector_store, doc_manager)

    if args.list_vertical is not None:
        vertical = None if args.list_vertical == "__ALL__" else args.list_vertical
        return list_documents(doc_manager, vertical=vertical)

    if args.stats:
        return print_stats(vector_store, doc_manager)

    if not args.vertical:
        parser.error("--vertical is required for ingestion")

    files = _resolve_files(args.files, args.directory)
    if not files:
        parser.error("Provide --files and/or --dir with at least one file")

    if args.sync:
        sync_collection(args.vertical, files, vector_store, doc_manager, base_dir=args.directory)
    else:
        ingest_files(args.vertical, files, vector_store, doc_manager, base_dir=args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
