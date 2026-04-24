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
_FOLDER_TERMS = {"folder", "folders", "directory", "directories", "dir", "section"}
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_DIGIT_WORDS = {value: key for key, value in _NUMBER_WORDS.items()}


def _query_tokens(query: str) -> List[str]:
    tokens = [
        token
        for token in _TOKEN_RE.findall((query or "").lower())
        if len(token) >= 2 or token.isdigit()
    ]
    meaningful = [token for token in tokens if token not in _STOPWORDS]
    return meaningful or tokens


def _expanded_tokens(tokens: List[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        if token in _NUMBER_WORDS:
            expanded.add(_NUMBER_WORDS[token])
        if token in _DIGIT_WORDS:
            expanded.add(_DIGIT_WORDS[token])
    return expanded


def _folder_query_tokens(tokens: List[str]) -> set[str]:
    folder_tokens: set[str] = set()
    for idx, token in enumerate(tokens):
        if token in _FOLDER_TERMS and idx + 1 < len(tokens):
            folder_tokens.add(tokens[idx + 1])
    if folder_tokens:
        return _expanded_tokens(list(folder_tokens))
    return set()


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
        clean.pop("_metadata_match", None)
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


def _where_for_doc_ids(doc_ids: List[str] | None) -> Dict | None:
    if doc_ids is None:
        return None
    clean_doc_ids = []
    seen = set()
    for doc_id in doc_ids:
        clean = str(doc_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        clean_doc_ids.append(clean)
    if not clean_doc_ids:
        return {"doc_id": "__no_rag_documents_selected__"}
    if len(clean_doc_ids) == 1:
        return {"doc_id": clean_doc_ids[0]}
    return {"doc_id": {"$in": clean_doc_ids}}


def _lexical_fallback_results(
    query: str,
    vector_store,
    collection_name: str,
    top_k: int,
    where: Dict | None = None,
) -> List[Dict]:
    tokens = _query_tokens(query)
    if not tokens:
        return []
    token_set = _expanded_tokens(tokens)
    folder_tokens = _folder_query_tokens(tokens)

    collection = vector_store.get_collection(collection_name)
    get_kwargs = {"include": ["documents", "metadatas"], "limit": 100000}
    if where:
        get_kwargs["where"] = where
    result = collection.get(**get_kwargs)
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    query_normalized = " ".join(tokens)
    ranked = []

    for doc, meta in zip(docs, metas):
        filename = str((meta or {}).get("filename", "") or "")
        source_path = str((meta or {}).get("source_path", "") or filename)
        folder_path = str((meta or {}).get("folder_path", "") or "")
        doc_text = str(doc or "")
        filename_lower = filename.lower()
        source_path_lower = source_path.lower()
        folder_path_lower = folder_path.lower()
        doc_lower = doc_text.lower()
        filename_tokens = set(_TOKEN_RE.findall(filename_lower))
        source_path_tokens = set(_TOKEN_RE.findall(source_path_lower))
        folder_path_tokens = set(_TOKEN_RE.findall(folder_path_lower))
        text_tokens = set(_TOKEN_RE.findall(doc_lower))

        overlap_filename = len(token_set & filename_tokens)
        overlap_source_path = len(token_set & source_path_tokens)
        overlap_folder_path = len(token_set & folder_path_tokens)
        overlap_text = len(token_set & text_tokens)
        phrase_bonus = 0.0
        metadata_match = ""
        if query_normalized and query_normalized in filename_lower:
            phrase_bonus += 2.0
            metadata_match = "filename"
        elif query_normalized and query_normalized in source_path_lower:
            phrase_bonus += 2.0
            metadata_match = "source_path"
        elif query_normalized and folder_path_lower and query_normalized in folder_path_lower:
            phrase_bonus += 2.5
            metadata_match = "folder"
        elif len(tokens) >= 2:
            joined_pairs = [" ".join(tokens[idx: idx + 2]) for idx in range(len(tokens) - 1)]
            if any(pair in filename_lower for pair in joined_pairs):
                phrase_bonus += 1.25
                metadata_match = "filename"
            elif any(pair in source_path_lower for pair in joined_pairs):
                phrase_bonus += 1.25
                metadata_match = "source_path"
            elif folder_path_lower and any(pair in folder_path_lower for pair in joined_pairs):
                phrase_bonus += 1.5
                metadata_match = "folder"

        if folder_tokens and folder_tokens & _expanded_tokens(list(folder_path_tokens)):
            phrase_bonus += 3.0
            metadata_match = "folder"
        elif folder_tokens and folder_tokens & _expanded_tokens(list(source_path_tokens)):
            phrase_bonus += 2.0
            metadata_match = "source_path"

        if query_normalized and query_normalized in doc_lower:
            phrase_bonus += 0.75

        lexical_score = (
            phrase_bonus
            + (1.2 * overlap_filename / max(len(tokens), 1))
            + (1.5 * overlap_source_path / max(len(tokens), 1))
            + (2.0 * overlap_folder_path / max(len(tokens), 1))
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
                "_metadata_match": metadata_match,
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


def _should_prefer_metadata_match(
    query: str,
    semantic_results: List[Dict],
    lexical_results: List[Dict],
    min_score: float,
) -> str:
    if not lexical_results:
        return ""

    top_lexical = lexical_results[0]
    top_meta = top_lexical.get("metadata", {})
    filename_lower = str(top_meta.get("filename", "") or "").lower()
    source_path_lower = str(top_meta.get("source_path", "") or top_meta.get("filename", "") or "").lower()
    folder_path_lower = str(top_meta.get("folder_path", "") or "").lower()
    top_doc_id = top_meta.get("doc_id")
    tokens = _query_tokens(query)
    if not tokens or not top_doc_id:
        return ""

    normalized_query = " ".join(tokens)
    adjacent_pairs = [" ".join(tokens[idx: idx + 2]) for idx in range(len(tokens) - 1)]
    folder_tokens = _folder_query_tokens(tokens)
    folder_token_match = bool(folder_tokens and folder_tokens & _expanded_tokens(_TOKEN_RE.findall(folder_path_lower)))
    has_metadata_phrase = (
        normalized_query in filename_lower
        or normalized_query in source_path_lower
        or (folder_path_lower and normalized_query in folder_path_lower)
        or any(pair in filename_lower for pair in adjacent_pairs)
        or any(pair in source_path_lower for pair in adjacent_pairs)
        or (folder_path_lower and any(pair in folder_path_lower for pair in adjacent_pairs))
        or folder_token_match
    )
    if not has_metadata_phrase:
        return ""

    top_lexical_score = float(top_lexical.get("_lexical_score", top_lexical.get("score", 0.0)))
    other_doc_scores = [
        float(item.get("_lexical_score", item.get("score", 0.0)))
        for item in lexical_results[1:]
        if item.get("metadata", {}).get("doc_id") != top_doc_id
    ]
    lexical_gap = top_lexical_score - max(other_doc_scores, default=0.0)
    match_kind = str(top_lexical.get("_metadata_match") or "")
    if folder_token_match:
        match_kind = "folder"
    if top_lexical_score < 1.5 or (match_kind != "folder" and lexical_gap < 0.5):
        return ""

    if not semantic_results:
        return match_kind or "metadata"

    top_semantic = semantic_results[0]
    semantic_doc_id = top_semantic.get("metadata", {}).get("doc_id")
    semantic_score = float(top_semantic.get("score", 0.0))
    if semantic_doc_id == top_doc_id:
        return ""

    if match_kind == "folder":
        return "folder"

    return (match_kind or "metadata") if semantic_score < max(min_score + 0.15, 0.7) else ""


def retrieve_context(
    query: str,
    vector_store,
    collection_name: str,
    top_k: int = 5,
    min_score: float = 0.5,
    doc_ids: List[str] | None = None,
) -> List[Dict]:
    if not query.strip():
        return []
    if doc_ids is not None and not [doc_id for doc_id in doc_ids if str(doc_id or "").strip()]:
        return []

    query_embedding = get_embedding(query)
    where = _where_for_doc_ids(doc_ids)
    raw_results = vector_store.search(
        collection_name=collection_name,
        query_embedding=query_embedding,
        top_k=top_k,
        min_score=min_score,
        where=where,
    )
    semantic_results = _dedupe_results(raw_results, top_k)
    lexical_results = _lexical_fallback_results(query, vector_store, collection_name, top_k, where=where)
    metadata_match = _should_prefer_metadata_match(query, semantic_results, lexical_results, min_score)
    if metadata_match == "folder":
        return _strip_internal_fields(lexical_results)
    if metadata_match:
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
