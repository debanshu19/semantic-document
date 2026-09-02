"""Core lifecycle tests: draft -> finalize -> immutable -> search.

These run against the real embeddings pipeline, but forced onto the
built-in hashing backend (see app.embeddings) rather than
sentence-transformers -- that keeps the suite fast, deterministic and
network-independent, while still exercising genuine production code
(not a hand-rolled test double that could drift from reality).
"""
from __future__ import annotations

import pytest

from app import embeddings, sdoc
from app.embeddings import EmbeddingError


@pytest.fixture(autouse=True)
def force_hashing_backend(monkeypatch):
    monkeypatch.setattr(embeddings, "_load_backend", lambda: ("hashing", None))


def test_create_and_read_draft(tmp_path):
    path = sdoc.create_draft(tmp_path, "my-doc", title="My Doc", content="hello world")
    draft = sdoc.read_draft(path)
    assert draft.title == "My Doc"
    assert draft.content == "hello world"


def test_cannot_create_duplicate_name(tmp_path):
    sdoc.create_draft(tmp_path, "dup", title="Dup", content="x")
    with pytest.raises(sdoc.SDocError):
        sdoc.create_draft(tmp_path, "dup", title="Dup 2", content="y")


def test_finalize_produces_immutable_file_and_removes_draft(tmp_path):
    sdoc.create_draft(tmp_path, "doc1", title="Doc One", content="The quick brown fox jumps over the lazy dog.")
    final = sdoc.finalize_draft(tmp_path, "doc1")

    assert final.exists()
    assert not sdoc.draft_path(tmp_path, "doc1").exists()

    opened = sdoc.open_finalized(final)
    assert opened["meta"]["status"] == "FINALIZED"
    assert opened["meta"]["title"] == "Doc One"
    assert "quick brown fox" in opened["content"]


def test_opening_finalized_document_does_not_mutate_it(tmp_path):
    sdoc.create_draft(tmp_path, "doc2", title="Doc Two", content="Some immutable content here.")
    final = sdoc.finalize_draft(tmp_path, "doc2")

    before = final.read_bytes()
    sdoc.open_finalized(final)
    sdoc.open_finalized(final)
    sdoc.search(final, "immutable")
    after = final.read_bytes()

    assert before == after


def test_finalize_empty_draft_fails_safely(tmp_path):
    sdoc.create_draft(tmp_path, "doc3", title="Empty", content="   ")
    with pytest.raises(sdoc.SDocError):
        sdoc.finalize_draft(tmp_path, "doc3")
    # draft must still be there -- FAILED state, not corrupted state
    assert sdoc.draft_path(tmp_path, "doc3").exists()
    assert not sdoc.final_path(tmp_path, "doc3").exists()


def test_finalize_twice_raises(tmp_path):
    sdoc.create_draft(tmp_path, "doc4", title="Doc Four", content="content goes here")
    sdoc.finalize_draft(tmp_path, "doc4")
    with pytest.raises(sdoc.SDocError):
        sdoc.finalize_draft(tmp_path, "doc4")


def test_finalize_leaves_draft_intact_when_embedding_fails(tmp_path, monkeypatch):
    sdoc.create_draft(tmp_path, "doc5", title="Doc Five", content="some content to embed")

    def boom(texts):
        raise EmbeddingError("model unavailable")

    monkeypatch.setattr(sdoc, "embed_texts", boom)
    with pytest.raises(sdoc.SDocError):
        sdoc.finalize_draft(tmp_path, "doc5")

    assert sdoc.draft_path(tmp_path, "doc5").exists()
    assert not sdoc.final_path(tmp_path, "doc5").exists()


def test_search_ranks_relevant_chunk_first(tmp_path):
    content = (
        "Chapter about gardening.\n\n"
        "Tomatoes need plenty of sunlight and regular watering to thrive.\n\n"
        "Chapter about astronomy.\n\n"
        "Black holes warp spacetime so severely that not even light escapes.\n\n"
    )
    sdoc.create_draft(tmp_path, "doc6", title="Mixed Topics", content=content)
    final = sdoc.finalize_draft(tmp_path, "doc6")

    hits = sdoc.search(final, "tomatoes watering sunlight")
    assert hits
    assert "tomatoes" in hits[0].text.lower()

    hits2 = sdoc.search(final, "black holes spacetime")
    assert hits2
    assert "black holes" in hits2[0].text.lower()


def test_search_with_no_query_returns_empty(tmp_path):
    sdoc.create_draft(tmp_path, "doc7", title="Doc Seven", content="anything at all")
    final = sdoc.finalize_draft(tmp_path, "doc7")
    assert sdoc.search(final, "") == []
