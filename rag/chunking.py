"""Token-based text chunking utilities for local RAG ingestion."""

from __future__ import annotations

from typing import List

import tiktoken

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into token-based overlapping chunks."""
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) < MIN_CHUNK_SIZE:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < MIN_CHUNK_SIZE:
            break

        chunks.append(encoding.decode(chunk_tokens))

        if end >= len(tokens):
            break

    return chunks
