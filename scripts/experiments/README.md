# Experiments

Standalone, throwaway scripts for evaluating architectural alternatives.
Nothing here is imported by the app -- these exist purely to generate
concrete evidence for design decisions, not to be maintained long-term.

## `compare_milvus.py` -- Milvus Lite vs. brute-force numpy

**Question:** would swapping our brute-force cosine similarity search for
Milvus (a real vector database) improve search accuracy or performance?

**Method:** identical chunking (`app.chunking`) and identical embeddings
(`app.embeddings`, the real sentence-transformers model) fed into both
approaches -- Milvus doesn't generate embeddings itself, so the only
variable under test is the vector search backend. A realistic
11-chunk, 6-topic document (storage choices, hotspots, consistency
models, caching, API design) was searched with 5 natural-language
queries against both:

- **Approach A**: our current approach -- brute-force numpy dot product
  over vectors stored directly in the `.sdoc` SQLite file.
- **Approach B**: Milvus Lite (embedded, file-based, no server) --
  create a collection, insert the same vectors, build its ANN index,
  query via `MilvusClient.search()`.

### Results (run on 2026-09-03)

**Ranking accuracy: identical.** All 5 queries returned the exact same
top-3 chunks in the exact same order, with the exact same similarity
scores, from both approaches. At this scale, Milvus's "approximate"
nearest-neighbor index is, in practice, exact -- there's no accuracy
upside to gain.

**Per-query search latency: comparable**, both in the single-digit to
low-hundreds of milliseconds range (dominated by embedding the query
text, not by the vector comparison itself, which is trivial at 11
vectors either way).

**One-time setup cost per document open: 5,295.64ms** to create a
Milvus collection, insert 11 vectors, and build its index. This is the
concrete cost of the "load the document's vectors onto Milvus every
time it's opened" flow that was proposed -- over five seconds of dead
time before a single search can run, for a document our current
approach opens and searches in effectively zero setup time (it's
already sitting in a SQLite file you just read).

### Conclusion

No accuracy improvement, no meaningful performance improvement for
in-document search, and a very real ~5 second tax on every document
open if vectors have to be freshly loaded into a Milvus collection each
time (per the originally proposed flow). Confirms the architectural
reasoning in the main README's "Why not a vector database?" section
with actual measurements rather than just theory.

This would look different at a much larger scale -- e.g. a hypothetical
"search across my entire library of hundreds of documents at once"
feature, where an index amortized *once* (not rebuilt per file) might
start to pay for itself. Even then, an in-process option like
`sqlite-vec` or FAISS would fit this project's local-first,
no-external-service architecture better than a Milvus deployment would.

### Reproducing

```bash
uv pip install pymilvus milvus-lite --index-url <your pypi mirror>
PYTHONPATH=. uv run python scripts/experiments/compare_milvus.py
```

Not added to `pyproject.toml` -- same reasoning as
`sentence-transformers`/torch: these are large, optional dependencies
that shouldn't be part of every `uv sync` just to keep this experiment
re-runnable.
