"""Tests against the real (unpatched) reranker module.

Unlike test_embeddings.py, a missing reranker isn't an error case to
assert on -- rerank() is designed to degrade gracefully (see
app/reranker.py's docstring for why: search is read-only against
already-committed data, so there's nothing to protect by failing hard).
So this file just verifies: if the real cross-encoder IS available in
this environment (it is, per our sentence-transformers install), it
actually produces sane, query-relevant rankings.
"""
from __future__ import annotations

import pytest

from app import reranker


@pytest.mark.skipif(not reranker.available(), reason="cross-encoder not installed in this environment")
def test_rerank_scores_relevant_candidate_higher():
    reranker._get_model.cache_clear()
    scores = reranker.rerank(
        "what are the database choices",
        [
            "The weather today is sunny with a light breeze.",
            "For storage, choose between a real-time store, a durable store, and an analytics store.",
        ],
    )
    assert scores is not None
    assert len(scores) == 2
    assert scores[1] > scores[0]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_rerank_empty_candidates_returns_empty_list():
    assert reranker.rerank("anything", []) == []


def test_rerank_returns_none_on_failure(monkeypatch):
    def boom():
        raise RuntimeError("model exploded")

    monkeypatch.setattr(reranker, "_get_model", boom)
    assert reranker.rerank("query", ["some candidate text"]) is None
