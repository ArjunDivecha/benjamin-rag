from rag.embeddings import EMBEDDING_DIM, get_embedding, get_embedding_model, get_embeddings_batch


def test_get_embedding_shape_and_type():
    vec = get_embedding("hello")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec)


def test_get_embeddings_batch_shape():
    vectors = get_embeddings_batch(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_same_text_is_deterministic():
    v1 = get_embedding("deterministic embedding test")
    v2 = get_embedding("deterministic embedding test")
    diffs = [abs(a - b) for a, b in zip(v1, v2)]
    assert max(diffs) < 1e-6


def test_different_texts_are_different():
    v1 = get_embedding("semiconductors market dynamics")
    v2 = get_embedding("healthcare provider reimbursement pathways")
    diffs = [abs(a - b) for a, b in zip(v1, v2)]
    assert sum(diffs) > 1e-3


def test_embedding_model_singleton():
    m1 = get_embedding_model()
    m2 = get_embedding_model()
    assert m1 is m2
