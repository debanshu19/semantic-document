"""Chunking is pure and has no external deps -- test it directly."""
from app.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_a_single_chunk():
    text = "Hello there, this is a short document."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)


def test_offsets_trace_back_to_original_text():
    text = "para one. " * 50 + "\n\n" + "para two. " * 50
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        # every chunk's recorded text must be extractable via its own offsets
        assert text[chunk.start:chunk.end].strip() == chunk.text


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
