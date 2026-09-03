"""Cross-encoder reranking: a second-stage refinement over whatever the
first-stage hybrid (keyword + semantic) retrieval already narrowed down.

Why this exists: app.embeddings uses a *bi-encoder* -- it embeds the
query and each chunk independently, then compares vectors. That's fast
enough to run over every chunk in a document, but the model never
actually sees the query and the chunk together, so it can miss
word-order and interaction nuances that decide real relevance.
Cross-encoders fix that by scoring (query, chunk) pairs jointly --
meaningfully more accurate, but too slow to run over every chunk in a
large corpus. The standard pattern (and what app.sdoc.search does):
use the cheap bi-encoder + keyword search to narrow a candidate pool
down to a shortlist, then spend the cross-encoder's accuracy only on
that shortlist.

This uses the same `sentence-transformers` package as app.embeddings --
CrossEncoder ships as part of it, so there's no new dependency, just
one more (small, ~90MB) model to cache via scripts/install-embeddings.sh.

Design difference from app.embeddings on purpose: a missing embedding
model is fatal for *finalize* (see the design doc's FAILED state --
nothing half-built should ever get published). A missing reranker at
*search* time is not treated the same way here -- search runs read-only
against a document that's already safely committed, so there's nothing
to protect by failing hard. rerank() returns None instead of raising,
and callers fall back to the first-stage ranking.
"""
from __future__ import annotations

import functools
import logging
import os

import numpy as np

# Must be set before `import sentence_transformers` pulls in its
# networking code -- same reasoning as app/embeddings.py. Set here too
# (not just there) because this module must work correctly even if
# something imports it *without* having imported app.embeddings first --
# relying on import order for a correctness property like "never phones
# home" is exactly the kind of subtle bug that bites later.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def available() -> bool:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: list[str]) -> list[float] | None:
    """Scores each (query, candidate) pair, higher is more relevant.

    Returns a list of scores in (0, 1), aligned 1:1 with `candidates`.
    Returns None -- never raises -- if the reranker isn't available or
    fails for any reason; callers should fall back to their own ranking
    in that case.
    """
    if not candidates:
        return []
    try:
        model = _get_model()
        raw_scores = model.predict([(query, c) for c in candidates])
    except Exception:  # noqa: BLE001 - reranking is a nice-to-have, never fatal
        logger.warning("Reranker unavailable; falling back to first-stage ranking.", exc_info=True)
        return None
    # Cross-encoder outputs are raw logits, not bounded to a fixed range.
    # Squash through a sigmoid for a consistent, interpretable 0-1 scale
    # that the UI can display alongside the keyword/semantic scores.
    return [float(1.0 / (1.0 + np.exp(-s))) for s in raw_scores]
