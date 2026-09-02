"""Local embedding generation via sentence-transformers. No cloud calls,
ever -- Phase 1's privacy model depends on that (see docs/design.md
section 9).

There is deliberately exactly one embedding model. The model is
downloaded from Hugging Face on first use and cached under
~/.cache/torch/sentence_transformers -- after that it's fully offline.

If it can't be loaded (not installed, no network for the first
download, etc.), embed_texts raises EmbeddingError. app.sdoc treats that
as fatal-but-safe: the draft stays untouched and no finalized artifact
is published -- this is the design doc's FAILED state, working exactly
as intended rather than silently degrading to a weaker model.
"""
from __future__ import annotations

import functools

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingError(Exception):
    """Raised when embedding generation fails outright. Finalization must
    treat this as fatal-but-safe: the draft stays untouched, no finalized
    artifact is published."""


@functools.lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via EmbeddingError path
        raise EmbeddingError(
            "sentence-transformers is not installed. Run "
            "`uv pip install sentence-transformers` (requires network access "
            "to Hugging Face on first use to download the model)."
        ) from exc

    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal for finalize
        raise EmbeddingError(
            f"Could not load embedding model '{MODEL_NAME}'. If this is the "
            "first run, the model needs to download from Hugging Face once -- "
            f"check your network/proxy settings. Original error: {exc}"
        ) from exc


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings. Returns float32 array of shape (n, EMBEDDING_DIM)."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    try:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"sentence-transformers encoding failed: {exc}") from exc
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query string. Returns float32 vector of shape (EMBEDDING_DIM,)."""
    return embed_texts([query])[0]
