"""Core lifecycle tests: draft -> finalize -> immutable -> search.

Embeddings are monkeypatched with a deterministic test double (see
tests/conftest.py) so these tests stay fast, offline and reproducible
without needing the real sentence-transformers model installed.
"""
from __future__ import annotations

import pytest

from app import sdoc
from app.embeddings import EmbeddingError


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


def test_search_natural_language_query_still_gets_keyword_signal(tmp_path):
    """Regression test for the implicit-AND FTS5 bug: a plain space-
    separated MATCH query requires *every* word in the same chunk, so a
    natural-language query with common words (the/a/is/...) almost never
    matched anything, silently zeroing out the keyword signal entirely.
    _build_fts_query ORs the terms together instead."""
    sdoc.create_draft(
        tmp_path, "doc11", title="Doc Eleven",
        content="The database choices for this system include a real-time store and a durable store.",
    )
    final = sdoc.finalize_draft(tmp_path, "doc11")

    hits = sdoc.search(final, "what are the database choices")
    assert hits
    assert hits[0].keyword_score > 0.0


def test_search_falls_back_gracefully_when_reranker_unavailable(tmp_path, monkeypatch):
    """Reranking is a nice-to-have at search time (unlike embeddings at
    finalize time) -- if it's unavailable, search must still work, just
    ranked by the first-stage fused score instead of a rerank score."""
    monkeypatch.setattr(sdoc, "cross_encoder_rerank", lambda query, candidates: None)

    content = "Tomatoes need sunlight.\n\nBlack holes warp spacetime."
    sdoc.create_draft(tmp_path, "doc12", title="Doc Twelve", content=content)
    final = sdoc.finalize_draft(tmp_path, "doc12")

    hits = sdoc.search(final, "tomatoes sunlight")
    assert hits
    assert hits[0].rerank_score is None
    assert "tomatoes" in hits[0].text.lower()


def test_reciprocal_rank_fusion_rewards_agreement_between_rankings():
    keyword = {1: 10.0, 2: 5.0, 3: 1.0}
    semantic = {2: 0.9, 1: 0.5, 3: 0.1}
    fused = sdoc._reciprocal_rank_fusion(keyword, semantic)
    # chunk 2 is #2 in keyword and #1 in semantic -- best combined agreement
    assert fused[2] >= fused[1]
    assert fused[1] >= fused[3]


def test_finalize_produces_exactly_one_file_no_sidecars(tmp_path):
    """The whole point of .sdoc is one portable artifact -- no -wal/-shm
    journal sidecars left behind (a real bug caught during manual testing:
    WAL journal mode leaves exactly those files sitting next to the db)."""
    sdoc.create_draft(tmp_path, "doc8", title="Doc Eight", content="single file, no sidecars please")
    sdoc.finalize_draft(tmp_path, "doc8")

    produced = {p.name for p in tmp_path.iterdir()}
    assert produced == {"doc8.sdoc"}


def test_finalize_to_explicit_output_path(tmp_path):
    """Simulates finalizing to a location the user picked via a native
    Save dialog -- somewhere entirely outside the library dir."""
    library_dir = tmp_path / "library"
    elsewhere = tmp_path / "elsewhere" / "my-chosen-name.sdoc"

    sdoc.create_draft(library_dir, "doc9", title="Doc Nine", content="saved somewhere the user picked")
    result = sdoc.finalize_draft(library_dir, "doc9", output_path=elsewhere)

    assert result == elsewhere
    assert elsewhere.exists()
    assert not sdoc.draft_path(library_dir, "doc9").exists()
    assert not sdoc.final_path(library_dir, "doc9").exists()  # nothing left in the library dir
    opened = sdoc.open_finalized(elsewhere)
    assert opened["content"] == "saved somewhere the user picked"


def test_finalize_refuses_to_overwrite_existing_output_path(tmp_path):
    library_dir = tmp_path / "library"
    target = tmp_path / "already-there.sdoc"
    target.write_text("not a real sdoc, just occupying the path")

    sdoc.create_draft(library_dir, "doc10", title="Doc Ten", content="content")
    with pytest.raises(sdoc.SDocError):
        sdoc.finalize_draft(library_dir, "doc10", output_path=target)
    assert sdoc.draft_path(library_dir, "doc10").exists()


def test_search_with_no_query_returns_empty(tmp_path):
    sdoc.create_draft(tmp_path, "doc7", title="Doc Seven", content="anything at all")
    final = sdoc.finalize_draft(tmp_path, "doc7")
    assert sdoc.search(final, "") == []
