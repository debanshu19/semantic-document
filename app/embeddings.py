"""Local embedding generation. No cloud calls, ever -- Phase 1's privacy
model depends on that (see docs/design.md section 9).

Two backends, selected automatically and cached for the process:

- "sentence-transformers": the real local transformer model. Best
  quality. Downloaded from Hugging Face on first use and cached under
  ~/.cache/torch/sentence_transformers -- after that it's fully offline.
- "hashing": a pure-numpy feature-hashed bag-of-words embedder. Zero
  extra dependencies, zero downloads, works instantly everywhere. Lower
  semantic quality (it rewards lexical overlap more than true meaning)
  but a perfectly honest MVP fallback when the ML stack isn't installed
  or reachable (e.g. restricted networks, air-gapped dev boxes).

Whichever backend actually produced a document's embeddings gets
recorded in that document's metadata (see app.sdoc), so .sdoc files stay
self-describing and portable regardless of which backend built them.
"""
from __future__ import annotations

import functools
import logging
import re
import zlib

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HASHING_MODEL_NAME = "hashing-bow-v1"

_WORD_RE = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """Raised when embedding generation fails outright. Finalization must
    treat this as fatal-but-safe: the draft stays untouched, no finalized
    artifact is published."""


@functools.lru_cache(maxsize=1)
def _load_backend():
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(ST_MODEL_NAME)
        logger.info("Embedding backend: sentence-transformers (%s)", ST_MODEL_NAME)
        return ("sentence-transformers", model)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
        logger.warning(
            "sentence-transformers unavailable (%s). Falling back to the "
            "built-in hashing embedder -- search will still work, just with "
            "lower semantic quality than the real model.",
            exc,
        )
        return ("hashing", None)


def current_model_name() -> str:
    """Whichever backend is actually active right now -- gets stamped into
    a document's metadata at finalize time."""
    backend, _ = _load_backend()
    return ST_MODEL_NAME if backend == "sentence-transformers" else HASHING_MODEL_NAME


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings. Returns float32 array of shape (n, EMBEDDING_DIM)."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    backend, model = _load_backend()
    if backend == "sentence-transformers":
        try:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"sentence-transformers encoding failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32)

    return _hashing_embed(texts)


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query string. Returns float32 vector of shape (EMBEDDING_DIM,)."""
    return embed_texts([query])[0]


def _hashing_embed(texts: list[str]) -> np.ndarray:
    """Deterministic, dependency-free fallback embedder: feature-hashed,
    L2-normalized bag-of-words. Uses zlib.crc32 rather than Python's
    built-in hash() -- str hashing is randomized per-process by default,
    which would silently break search the moment the app restarts between
    finalizing a document and querying it. crc32 is stable across
    processes, machines and Python versions, which matters a lot for a
    file format whose whole point is portability."""
    vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in _WORD_RE.findall(text.lower()):
            bucket = zlib.crc32(word.encode("utf-8")) % EMBEDDING_DIM
            vectors[i, bucket] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors
