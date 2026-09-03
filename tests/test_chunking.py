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


def test_default_chunk_size_favors_smaller_more_focused_chunks():
    from app.chunking import DEFAULT_CHUNK_SIZE

    # Smaller chunks keep each chunk's embedding focused on one idea --
    # a deliberate accuracy improvement, not just an arbitrary number.
    assert DEFAULT_CHUNK_SIZE <= 500


def test_prefers_sentence_boundary_over_hard_cut():
    sentence_a = "The quick brown fox jumps over the lazy dog" * 3 + ". "
    sentence_b = "A completely different sentence about something else entirely" * 3 + ". "
    text = sentence_a + sentence_b
    chunks = chunk_text(text, chunk_size=len(sentence_a) + 20, overlap=10)
    # The first chunk should end right at (or very near) the sentence
    # boundary, not mid-word into sentence_b.
    assert chunks[0].text.endswith(".")


def test_prefers_paragraph_over_sentence_boundary():
    para_a = "First paragraph with some words in it that goes on a bit."
    para_b = "Second paragraph that also has a decent amount of text in it."
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text, chunk_size=len(para_a) + 5, overlap=10)
    assert chunks[0].text == para_a
