"""Shared pytest fixtures.

The app ships exactly one embedding model (sentence-transformers) and
one cross-encoder reranker -- see app/embeddings.py and app/reranker.py.
Downloading/running the real models in every test run would be slow,
network-dependent, and non-deterministic, so tests monkeypatch both with
small deterministic test doubles instead. This is a test-only concern;
it has no bearing on how the app behaves in production.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from app import sdoc

_FAKE_EMBEDDING_DIM = 384


def _fake_embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Deterministic feature-hashed bag-of-words -- good enough to make
    search-ranking assertions meaningful in tests, without needing the
    real model. Uses zlib.crc32 (stable across processes), not Python's
    randomized-per-process hash().

    Accepts (and ignores) `model_name` so tests exercising the
    multi-model registry can use the same fake for any model choice --
    the important properties (deterministic, offline, meaningful for
    ranking) don't depend on which model was picked."""
    vectors = np.zeros((len(texts), _FAKE_EMBEDDING_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            bucket = zlib.crc32(word.encode("utf-8")) % _FAKE_EMBEDDING_DIM
            vectors[i, bucket] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


def _fake_rerank(query: str, candidates: list[str]) -> list[float]:
    """Deterministic word-overlap scorer, squashed to (0, 1) -- stands in
    for the real cross-encoder so reranking-dependent tests stay fast,
    offline and reproducible."""
    query_words = set(query.lower().split())
    scores = []
    for text in candidates:
        overlap = len(query_words & set(text.lower().split()))
        scores.append(overlap / (overlap + 1))
    return scores


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(sdoc, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(sdoc, "embed_query", lambda q, model_name=None: _fake_embed_texts([q])[0])


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch):
    monkeypatch.setattr(sdoc, "cross_encoder_rerank", _fake_rerank)
