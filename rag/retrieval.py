"""Retrieval and context assembly for local RAG queries."""

from __future__ import annotations

from typing import Dict, List

from rag.embeddings import get_embedding


def retrieve_context(
    query: str,
    vector_store,
    collection_name: str,
    top_k: int = 5,
    min_score: float = 0.5,
) -> List[Dict]:
    if not query.strip():
        return []

    query_embedding = get_embedding(query)
    raw_results = vector_store.search(
        collection_name=collection_name,
        query_embedding=query_embedding,
        top_k=top_k,
        min_score=min_score,
    )

    deduped = []
    seen = set()
    for item in raw_results:
        meta = item.get("metadata", {})
        key = (meta.get("doc_id"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:top_k]


def assemble_context(chunks: List[Dict]) -> str:
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        chunk_id = meta.get("chunk_id", 0)
        total_chunks = meta.get("total_chunks", "?")
        text = chunk.get("text", "").strip()
        parts.append(
            f'<document name="{filename}" chunk="{chunk_id}/{total_chunks}">\n'
            f"{text}\n"
            f"</document>"
        )

    return "\n\n".join(parts)
