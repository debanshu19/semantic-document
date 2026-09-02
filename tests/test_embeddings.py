"""Tests against the real (unpatched) embeddings module -- this is the
one place we deliberately do NOT use the conftest.py fake, because the
whole point is verifying what actually happens with the single real
backend.

In this dev environment sentence-transformers isn't installed (it's a
large optional dependency -- see README), so this doubles as a real
regression test for the "one model, fails safely instead of silently
degrading" behavior itself.
"""
from __future__ import annotations

import importlib

import pytest

from app import embeddings


def test_embed_texts_raises_clear_error_when_model_unavailable():
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed in this environment")
    except ImportError:
        pass

    embeddings._get_model.cache_clear()
    with pytest.raises(embeddings.EmbeddingError, match="sentence-transformers"):
        embeddings.embed_texts(["some text to embed"])


def test_embed_texts_empty_list_short_circuits_without_loading_model():
    importlib.reload(embeddings)  # fresh, unmemoized state
    result = embeddings.embed_texts([])
    assert result.shape == (0, embeddings.EMBEDDING_DIM)
