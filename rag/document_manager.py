"""Document metadata and local file lifecycle manager."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class DocumentManager:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "metadata.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    vertical TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    stored_path TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "source_path" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN source_path TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                UPDATE documents
                SET source_path = filename
                WHERE source_path IS NULL OR TRIM(source_path) = ''
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_vertical ON documents(vertical)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)"
            )

    def _content_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def _doc_id(self, content_hash: str, source_path: str) -> str:
        identity = f"{source_path}\0{content_hash}".encode("utf-8")
        return f"doc_{hashlib.sha256(identity).hexdigest()[:12]}"

    def _file_type(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return suffix.lstrip(".") if suffix else "unknown"

    def _normalize_source_path(self, filename: str, source_path: Optional[str] = None) -> str:
        raw_path = str(source_path or filename or "").replace("\\", "/").strip()
        parts = [
            part
            for part in raw_path.strip("/").split("/")
            if part and part not in {".", ".."}
        ]
        normalized = "/".join(parts)
        return normalized or os.path.basename(filename) or "untitled"

    def _add_derived_fields(self, doc: Dict) -> Dict:
        source_path = self._normalize_source_path(
            str(doc.get("filename", "")),
            str(doc.get("source_path") or doc.get("filename") or ""),
        )
        doc["source_path"] = source_path
        doc["folder_path"] = str(Path(source_path).parent).replace("\\", "/")
        if doc["folder_path"] == ".":
            doc["folder_path"] = ""
        return doc

    def _stored_filename(self, doc_id: str, filename: str) -> str:
        safe_name = os.path.basename(filename).replace("/", "_")
        return f"{doc_id}_{safe_name}"

    def save_document(
        self,
        file_bytes: bytes,
        filename: str,
        vertical: str,
        chunk_count: int = 0,
        source_path: Optional[str] = None,
    ) -> str:
        content_hash = self._content_hash(file_bytes)
        normalized_source_path = self._normalize_source_path(filename, source_path)
        doc_id = self._doc_id(content_hash, normalized_source_path)

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE source_path = ? AND content_hash = ? AND vertical = ?
                LIMIT 1
                """,
                (normalized_source_path, content_hash, vertical),
            ).fetchone()
            if existing:
                return str(existing["doc_id"])

            existing = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if existing:
                return doc_id

            stored_name = self._stored_filename(doc_id, filename)
            stored_path = self.storage_path / stored_name
            stored_path.write_bytes(file_bytes)

            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, source_path, vertical, file_type, file_size,
                    chunk_count, content_hash, ingested_at, stored_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    filename,
                    normalized_source_path,
                    vertical,
                    self._file_type(filename),
                    len(file_bytes),
                    chunk_count,
                    content_hash,
                    datetime.now(timezone.utc).isoformat(),
                    str(stored_path),
                ),
            )

        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return self._add_derived_fields(dict(row)) if row else None

    def list_documents(self, vertical: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM documents"
        params = ()
        if vertical:
            query += " WHERE vertical = ?"
            params = (vertical,)
        query += " ORDER BY ingested_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._add_derived_fields(dict(r)) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        doc = self.get_document(doc_id)
        if not doc:
            return

        stored_path = Path(doc["stored_path"])
        if stored_path.exists():
            stored_path.unlink()

        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def is_unchanged(
        self,
        file_bytes: bytes,
        filename: str,
        vertical: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> bool:
        content_hash = self._content_hash(file_bytes)
        normalized_source_path = self._normalize_source_path(filename, source_path)
        query = """
            SELECT 1
            FROM documents
            WHERE source_path = ? AND content_hash = ?
        """
        params: tuple = (normalized_source_path, content_hash)
        if vertical is not None:
            query += " AND vertical = ?"
            params = (normalized_source_path, content_hash, vertical)
        query += " LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return row is not None
