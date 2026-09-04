"""Local embedding generation via sentence-transformers. No cloud calls,
ever -- Phase 1's privacy model depends on that (see docs/design.md
section 9).

A finalized .sdoc records the exact embedding model used to build it (in
the meta table). At search time we reload *that specific model* to embed
the query -- otherwise the query vector would sit in a different vector
space than the stored chunk vectors and cosine similarity would be
meaningless. That's why finalize_draft accepts a model choice and search
routes it via the meta record automatically.

Model weights are cached under ~/.cache/huggingface the first time
they're fetched (scripts/install-embeddings.sh handles the one-time
download; pass model names as arguments to grab non-default ones). After
that, this module runs in Hugging Face's offline mode -- it will NEVER
make a network call on its own. That's not just a corporate-proxy
workaround; it's the correct behavior for an app whose entire privacy
model depends on "no cloud calls, ever." If a model isn't cached yet,
that's a setup step to run once, not something finalize should silently
reach out to the internet for on every request.

If a model can't be loaded (not installed, or not cached yet),
embed_texts raises EmbeddingError. app.sdoc treats that as fatal-but-safe:
the draft stays untouched and no finalized artifact is published -- this
is the design doc's FAILED state, working exactly as intended rather
than silently degrading to a weaker model.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np

# Must be set before `import sentence_transformers` / `import huggingface_hub`
# pulls in its networking code -- see module docstring.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@dataclass(frozen=True)
class ModelSpec:
    """One curated embedding model that the app knows how to use.

    Kept as an explicit registry rather than accepting arbitrary
    user-supplied model strings, because (a) any model with a wildly
    different tokenizer/output shape needs verification against our
    pipeline, and (b) the UI needs to show human-meaningful choices
    (size, speed, quality trade-off) rather than raw Hub paths.
    """

    name: str            # short id stored in .sdoc meta -- the stable contract
    hf_path: str         # what actually gets passed to SentenceTransformer(...)
    dim: int             # vector dimensionality (embedding output size)
    size_mb: int         # rough on-disk size for the UI to show
    description: str


# Order here is the order shown in the UI dropdown. Default is first.
_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="all-MiniLM-L6-v2",
        hf_path="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        size_mb=80,
        description="Small and fast. Solid general-purpose baseline. Recommended default.",
    ),
    ModelSpec(
        name="all-mpnet-base-v2",
        hf_path="sentence-transformers/all-mpnet-base-v2",
        dim=768,
        size_mb=420,
        description="Higher quality, larger and slower. Bigger vectors -> bigger .sdoc files.",
    ),
    ModelSpec(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        hf_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
        size_mb=470,
        description="Multilingual (50+ languages). Use for non-English or mixed-language documents.",
    ),
)
_REGISTRY_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in _REGISTRY}
# Backward-compat: older .sdoc files stored the full HF path in meta.model_name
# rather than the short name; keep an alias table so open/search still work.
_REGISTRY_BY_HF_PATH: dict[str, ModelSpec] = {m.hf_path: m for m in _REGISTRY}

DEFAULT_MODEL_NAME: str = _REGISTRY[0].name


class EmbeddingError(Exception):
    """Raised when embedding generation fails outright. Finalization must
    treat this as fatal-but-safe: the draft stays untouched, no finalized
    artifact is published."""


def available_models() -> tuple[ModelSpec, ...]:
    """Public registry view for the UI to render as dropdown options."""
    return _REGISTRY


def resolve_model(name: str | None) -> ModelSpec:
    """Look up a ModelSpec by short name, or by full HF path (for backward
    compatibility with .sdoc files finalized before the registry existed).
    None -> default."""
    if name is None or name == "":
        return _REGISTRY_BY_NAME[DEFAULT_MODEL_NAME]
    if name in _REGISTRY_BY_NAME:
        return _REGISTRY_BY_NAME[name]
    if name in _REGISTRY_BY_HF_PATH:
        return _REGISTRY_BY_HF_PATH[name]
    raise EmbeddingError(
        f"Unknown embedding model '{name}'. This .sdoc was likely finalized "
        f"with a model this build of the app does not know about. Known "
        f"models: {sorted(_REGISTRY_BY_NAME)}"
    )


@functools.lru_cache(maxsize=None)
def _get_model(hf_path: str):
    """Cached per-model loader. `lru_cache(maxsize=None)` keeps every model
    the process ever uses in memory -- that's fine and desirable: models
    are a few hundred MB max, and if a user opens two documents finalized
    with different models in one session, we don't want to pay the load
    cost twice."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via EmbeddingError path
        raise EmbeddingError(
            "sentence-transformers is not installed. Run "
            "./scripts/install-embeddings.sh to install it and cache the model."
        ) from exc

    try:
        return SentenceTransformer(hf_path)
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal for finalize
        raise EmbeddingError(
            f"Could not load embedding model '{hf_path}'. It needs to be "
            f"downloaded and cached once before first use -- run "
            f"./scripts/install-embeddings.sh {hf_path.split('/')[-1]} "
            f"(this app runs fully offline otherwise and will not fetch it "
            f"automatically). Original error: {exc}"
        ) from exc


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed a batch of strings using the named model (or the default).

    Returns float32 array of shape (n, spec.dim). The dimension varies by
    model -- callers relying on a specific dim should resolve the spec
    first via resolve_model()."""
    spec = resolve_model(model_name)
    if not texts:
        return np.zeros((0, spec.dim), dtype=np.float32)

    model = _get_model(spec.hf_path)
    try:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"sentence-transformers encoding failed: {exc}") from exc
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Embed a single search query string. Must use the same model the
    .sdoc was finalized with -- otherwise the query vector and the stored
    chunk vectors are in incompatible vector spaces."""
    return embed_texts([query], model_name=model_name)[0]
