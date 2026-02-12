"""Embedding utilities for local RAG using Sentence Transformers."""

from __future__ import annotations

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32

_MODEL: Optional[SentenceTransformer] = None


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and return a singleton embedding model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_select_device())
        _MODEL.eval()
    return _MODEL


def get_embedding(text: str) -> List[float]:
    """Embed a single text into a 768-dim vector."""
    model = get_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """Embed a list of texts into vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return vectors.tolist()
