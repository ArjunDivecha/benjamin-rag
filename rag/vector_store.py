"""Chroma vector store wrapper for local RAG collections."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import chromadb


def _similarity_from_distance(distance: float) -> float:
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class VectorStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)

    def _normalize_collection_name(self, name: str) -> str:
        clean = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
        if not clean:
            clean = "collection"
        if not clean[0].isalnum():
            clean = f"c{clean}"
        if not clean[-1].isalnum():
            clean = f"{clean}0"
        if len(clean) < 3:
            clean = f"col_{clean}"
        return clean

    def create_collection(self, name: str):
        internal_name = self._normalize_collection_name(name)
        return self.client.get_or_create_collection(
            name=internal_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection(self, name: str):
        internal_name = self._normalize_collection_name(name)
        return self.client.get_or_create_collection(name=internal_name)

    def upsert_document(
        self,
        collection_name: str,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict] = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        if not chunks:
            return

        collection = self.get_collection(collection_name)
        base_meta = dict(metadata or {})
        total_chunks = len(chunks)

        ids = []
        metadatas = []
        for idx, _ in enumerate(chunks):
            ids.append(f"{doc_id}:{idx}")
            chunk_meta = {
                **base_meta,
                "doc_id": doc_id,
                "chunk_id": idx,
                "total_chunks": total_chunks,
            }
            metadatas.append(chunk_meta)

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        collection = self.get_collection(collection_name)
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        result = collection.query(**query_kwargs)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        output = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = _similarity_from_distance(dist)
            if score < min_score:
                continue
            output.append(
                {
                    "text": doc,
                    "score": score,
                    "metadata": meta,
                }
            )
        return output

    def delete_document(self, collection_name: str, doc_id: str) -> None:
        collection = self.get_collection(collection_name)
        collection.delete(where={"doc_id": doc_id})

    def get_stats(self, collection_name: str) -> Dict:
        collection = self.get_collection(collection_name)
        total_vectors = collection.count()
        docs = self.list_documents(collection_name)
        return {
            "collection_name": collection_name,
            "vector_count": total_vectors,
            "doc_count": len(docs),
        }

    def list_documents(self, collection_name: str) -> List[Dict]:
        collection = self.get_collection(collection_name)
        result = collection.get(include=["metadatas"], limit=100000)
        metas = result.get("metadatas", []) or []

        by_doc = defaultdict(lambda: {"chunk_count": 0})
        for meta in metas:
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            entry = by_doc[doc_id]
            entry["doc_id"] = doc_id
            entry["filename"] = meta.get("filename")
            entry["source_path"] = meta.get("source_path") or meta.get("filename")
            entry["folder_path"] = meta.get("folder_path") or ""
            entry["chunk_count"] += 1

        return list(by_doc.values())
