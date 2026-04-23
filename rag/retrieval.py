"""Retrieval and context assembly for local RAG queries."""

from __future__ import annotations

import re
from typing import Dict, List

from rag.embeddings import get_embedding

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "doc", "docx", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "pdf", "please", "show",
    "summarize", "summarise", "that", "the", "this", "to", "what", "with",
}


def _query_tokens(query: str) -> List[str]:
    tokens = [token for token in _TOKEN_RE.findall((query or "").lower()) if len(token) >= 2]
    meaningful = [token for token in tokens if token not in _STOPWORDS]
    return meaningful or tokens


def _dedupe_results(items: List[Dict], top_k: int) -> List[Dict]:
    deduped = []
    seen = set()
    for item in items:
        meta = item.get("metadata", {})
        key = (meta.get("doc_id"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped


def _strip_internal_fields(items: List[Dict]) -> List[Dict]:
    cleaned = []
    for item in items:
        clean = dict(item)
        clean.pop("_lexical_score", None)
        cleaned.append(clean)
    return cleaned


def _top_doc_only(items: List[Dict], top_k: int) -> List[Dict]:
    if not items:
        return []
    top_doc_id = items[0].get("metadata", {}).get("doc_id")
    same_doc = [
        item
        for item in items
        if item.get("metadata", {}).get("doc_id") == top_doc_id
    ]
    return _dedupe_results(same_doc, top_k)


def _lexical_fallback_results(query: str, vector_store, collection_name: str, top_k: int) -> List[Dict]:
    tokens = _query_tokens(query)
    if not tokens:
        return []

    collection = vector_store.get_collection(collection_name)
    result = collection.get(include=["documents", "metadatas"], limit=100000)
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    query_normalized = " ".join(tokens)
    ranked = []

    for doc, meta in zip(docs, metas):
        filename = str((meta or {}).get("filename", "") or "")
        doc_text = str(doc or "")
        filename_lower = filename.lower()
        doc_lower = doc_text.lower()
        filename_tokens = set(_TOKEN_RE.findall(filename_lower))
        text_tokens = set(_TOKEN_RE.findall(doc_lower))

        overlap_filename = len(set(tokens) & filename_tokens)
        overlap_text = len(set(tokens) & text_tokens)
        phrase_bonus = 0.0
        if query_normalized and query_normalized in filename_lower:
            phrase_bonus += 2.0
        elif len(tokens) >= 2:
            joined_pairs = [" ".join(tokens[idx: idx + 2]) for idx in range(len(tokens) - 1)]
            if any(pair in filename_lower for pair in joined_pairs):
                phrase_bonus += 1.25

        if query_normalized and query_normalized in doc_lower:
            phrase_bonus += 0.75

        lexical_score = (
            phrase_bonus
            + (1.2 * overlap_filename / max(len(tokens), 1))
            + (0.5 * overlap_text / max(len(tokens), 1))
        )
        if lexical_score <= 0:
            continue

        ranked.append(
            {
                "text": doc_text,
                "score": min(0.99, 0.45 + lexical_score / 4.0),
                "metadata": meta,
                "_lexical_score": lexical_score,
            }
        )

    ranked.sort(
        key=lambda item: (
            item.get("_lexical_score", 0.0),
            item.get("score", 0.0),
            len(str(item.get("text", ""))),
        ),
        reverse=True,
    )
    return _dedupe_results(ranked, top_k)


def _should_prefer_filename_match(
    query: str,
    semantic_results: List[Dict],
    lexical_results: List[Dict],
    min_score: float,
) -> bool:
    if not lexical_results:
        return False

    top_lexical = lexical_results[0]
    top_meta = top_lexical.get("metadata", {})
    filename_lower = str(top_meta.get("filename", "") or "").lower()
    top_doc_id = top_meta.get("doc_id")
    tokens = _query_tokens(query)
    if not tokens or not top_doc_id:
        return False

    normalized_query = " ".join(tokens)
    adjacent_pairs = [" ".join(tokens[idx: idx + 2]) for idx in range(len(tokens) - 1)]
    has_filename_phrase = (
        normalized_query in filename_lower
        or any(pair in filename_lower for pair in adjacent_pairs)
    )
    if not has_filename_phrase:
        return False

    top_lexical_score = float(top_lexical.get("_lexical_score", top_lexical.get("score", 0.0)))
    other_doc_scores = [
        float(item.get("_lexical_score", item.get("score", 0.0)))
        for item in lexical_results[1:]
        if item.get("metadata", {}).get("doc_id") != top_doc_id
    ]
    lexical_gap = top_lexical_score - max(other_doc_scores, default=0.0)
    if top_lexical_score < 1.5 or lexical_gap < 0.5:
        return False

    if not semantic_results:
        return True

    top_semantic = semantic_results[0]
    semantic_doc_id = top_semantic.get("metadata", {}).get("doc_id")
    semantic_score = float(top_semantic.get("score", 0.0))
    if semantic_doc_id == top_doc_id:
        return False

    return semantic_score < max(min_score + 0.15, 0.7)


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
    semantic_results = _dedupe_results(raw_results, top_k)
    lexical_results = _lexical_fallback_results(query, vector_store, collection_name, top_k)
    if _should_prefer_filename_match(query, semantic_results, lexical_results, min_score):
        return _strip_internal_fields(_top_doc_only(lexical_results, top_k))
    if semantic_results:
        return _strip_internal_fields(semantic_results)

    return _strip_internal_fields(lexical_results)


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
