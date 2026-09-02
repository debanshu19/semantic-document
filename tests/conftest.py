"""Shared pytest fixtures.

The app ships exactly one embedding model (sentence-transformers) --
see app/embeddings.py. Downloading/running the real model in every test
run would be slow, network-dependent, and non-deterministic, so tests
monkeypatch the embed_texts/embed_query functions with a small
deterministic test double instead. This is a test-only concern; it has
no bearing on how the app behaves in production.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from app import sdoc

_FAKE_EMBEDDING_DIM = 384


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic feature-hashed bag-of-words -- good enough to make
    search-ranking assertions meaningful in tests, without needing the
    real model. Uses zlib.crc32 (stable across processes), not Python's
    randomized-per-process hash()."""
    vectors = np.zeros((len(texts), _FAKE_EMBEDDING_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            bucket = zlib.crc32(word.encode("utf-8")) % _FAKE_EMBEDDING_DIM
            vectors[i, bucket] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(sdoc, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(sdoc, "embed_query", lambda q: _fake_embed_texts([q])[0])
