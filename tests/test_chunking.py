import tiktoken

from rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE, chunk_text


ENC = tiktoken.get_encoding("cl100k_base")
UNIT = " a"  # 1 token per unit under cl100k_base


def _text_with_tokens(token_count: int) -> str:
    text = UNIT * token_count
    assert len(ENC.encode(text)) == token_count
    return text


def _expected_chunk_count(total_tokens: int, chunk_size: int, overlap: int) -> int:
    step = chunk_size - overlap
    count = 0
    for start in range(0, total_tokens, step):
        end = start + chunk_size
        chunk_len = min(end, total_tokens) - start
        if chunk_len < MIN_CHUNK_SIZE:
            break
        count += 1
        if end >= total_tokens:
            break
    return count


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_below_min_chunk_returns_empty_list():
    text = _text_with_tokens(MIN_CHUNK_SIZE - 1)
    assert chunk_text(text) == []


def test_exactly_chunk_size_returns_one_chunk():
    text = _text_with_tokens(CHUNK_SIZE)
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert len(ENC.encode(chunks[0])) == CHUNK_SIZE


def test_overlap_exists_for_1024_tokens():
    text = _text_with_tokens(1024)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert len(chunks) >= 2

    first_tokens = ENC.encode(chunks[0])
    second_tokens = ENC.encode(chunks[1])

    assert first_tokens[-CHUNK_OVERLAP:] == second_tokens[:CHUNK_OVERLAP]


def test_very_long_text_chunk_count_and_max_chunk_size():
    total_tokens = 5000
    text = _text_with_tokens(total_tokens)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    expected = _expected_chunk_count(total_tokens, CHUNK_SIZE, CHUNK_OVERLAP)
    assert len(chunks) == expected
    assert all(len(ENC.encode(c)) <= CHUNK_SIZE for c in chunks)


def test_round_trip_no_data_loss():
    total_tokens = 5000
    text = _text_with_tokens(total_tokens)
    original_tokens = ENC.encode(text)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    reconstructed = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = ENC.encode(chunk)
        if i == 0:
            reconstructed.extend(chunk_tokens)
        else:
            reconstructed.extend(chunk_tokens[CHUNK_OVERLAP:])

    assert reconstructed == original_tokens
