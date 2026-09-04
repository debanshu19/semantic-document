"""Tests against the real (unpatched) embeddings module -- this is the
one place we deliberately do NOT use the conftest.py fake, because the
whole point is verifying what actually happens with the real backend.

In this dev environment sentence-transformers may or may not be
installed (it's a large optional dependency -- see README), so these
also double as real regression tests for the "fail safely instead of
silently degrading" behavior itself.
"""
from __future__ import annotations

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
    # Deliberately does NOT reload the embeddings module -- reloading
    # would create a fresh EmbeddingError class object that no longer
    # matches sdoc.py's already-imported reference, breaking
    # `except EmbeddingError` elsewhere. Just verify the shape logic.
    default = embeddings.resolve_model(None)
    result = embeddings.embed_texts([])
    assert result.shape == (0, default.dim)


def test_resolve_model_by_short_name_returns_registry_entry():
    spec = embeddings.resolve_model("all-MiniLM-L6-v2")
    assert spec.name == "all-MiniLM-L6-v2"
    assert spec.dim == 384


def test_resolve_model_by_full_hf_path_still_works_for_legacy_sdocs():
    """Older .sdoc files (finalized before the model registry existed)
    stored the full HuggingFace path in meta.model_name, not the short
    name. Opening them after this refactor must still resolve correctly
    or search silently uses the wrong model / crashes."""
    spec = embeddings.resolve_model("sentence-transformers/all-MiniLM-L6-v2")
    assert spec.name == "all-MiniLM-L6-v2"


def test_resolve_model_with_none_returns_default():
    default = embeddings.resolve_model(None)
    assert default.name == embeddings.DEFAULT_MODEL_NAME


def test_resolve_model_unknown_name_raises_clear_error():
    with pytest.raises(embeddings.EmbeddingError, match="Unknown embedding model"):
        embeddings.resolve_model("some-model-we-never-heard-of")


def test_available_models_returns_non_empty_registry():
    models = embeddings.available_models()
    assert len(models) >= 1
    assert all(m.name and m.hf_path and m.dim > 0 for m in models)
    # Default is always the first entry (UI convention -- see registry docstring)
    assert models[0].name == embeddings.DEFAULT_MODEL_NAME
