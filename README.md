# Semantic Document

Write it. Finalize it. It's locked forever, and now it's semantically
searchable -- offline, with no external vector database required.

## The idea

Most "AI search over my notes" tools keep your text and its embeddings in
two separate places that can silently drift apart. Semantic Document
takes a different, deliberately strict approach for its first phase:
The combination is what's novel, not any single piece:
• File-first, not folder-first — the artifact is one document, not a vault or workspace
• Immutable snapshot — the searchable representation of the document at a moment in time is preserved forever (legally significant for contracts, evidence, published papers, sealed correspondence)
• Offline-first on mobile — no cloud, no account, no subscription; opens like a PDF
• Reader ecosystem — anyone can build a conformant reader on any platform (like PDF)


1. **Draft** -- write and edit freely, just like any text editor.
2. **Finalize** -- one action chunks the content, generates embeddings
   locally, builds a full-text + vector index, and commits everything
   into a single portable `.sdoc` file.
3. **Locked** -- that file is now immutable. Opening it never mutates it.
   Search reads directly from the indexes baked into the file. Want to
   change the content? Save it as a new document -- the original
   snapshot stays exactly as it was.

The `.sdoc` file is just a SQLite database under the hood: canonical
text, chunk records with source offsets, embedding vectors, an FTS5
keyword index, and integrity metadata -- all in one file (no `-wal`/
`-shm` sidecars) you can copy, back up, or hand to someone else, with no
server or database required to read it back.

See [`docs/design.md`](docs/design.md) for the full Phase 1 design
rationale (immutability, atomic finalization, privacy model, and what's
explicitly deferred to later phases).

## Running it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
./scripts/install-embeddings.sh   # the embedding model -- see below
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

Your documents live in `~/SemanticDocument/library/` -- not inside the
repo. That keeps them independent of wherever you happen to have checked
out the source, and safe from `git clean`. Override the location with
the `SDOC_LIBRARY_DIR` environment variable if you want it elsewhere.

Semantic Document ships a small **curated registry of embedding
models** you can pick from at finalize time -- a fast/small default,
a higher-quality larger option, and a multilingual one -- plus one
reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). All run fully
locally, so no document text or query ever leaves your machine. They're
kept as optional (rather than hard) dependencies only because they pull
in torch, a large download; functionally, they're the app's only
embedding and reranking backends, no fallback.

The **chosen embedding model is a permanent property of a finalized
.sdoc** -- once picked, the short name is written into the file's meta
table, and search always reloads *that specific model* to embed the
query. This isn't optional: if search embedded queries with a different
model than the stored chunks, the two vectors would live in
incompatible vector spaces and cosine similarity would be meaningless.
Older .sdoc files finalized before the registry existed (which stored
the full HuggingFace path) are still readable -- the resolver accepts
either form.

`scripts/install-embeddings.sh` handles the one-time setup: installs
the package and pre-downloads/caches models via the internal Hugging
Face mirror. It defaults to just the default embedding model + the
reranker (~170MB); pass model short names or `--all` to fetch extras:

```bash
./scripts/install-embeddings.sh                    # default only
./scripts/install-embeddings.sh all-mpnet-base-v2  # + higher-quality model
./scripts/install-embeddings.sh --all              # every registered model
```

After caching, the app runs everything in Hugging Face's **offline
mode** by default (`app/embeddings.py` and `app/reranker.py` each set
`HF_HUB_OFFLINE=1`) -- neither will ever make a network call on its
own. That's not just a corporate-network workaround; it's the correct
behavior for an app whose whole privacy model is "no cloud calls,
ever." If a picked model isn't cached yet, that's a setup step to run
once, not something search/finalize should silently reach out to the
internet for.

If the picked embedding model isn't installed/cached, finalize fails
safely and clearly: your draft is untouched, the error message spells
out exactly which model is missing and how to install it, and nothing
half-finished gets published. That's the design doc's intended FAILED
state working as designed, not a bug -- there's no silent degrading to
a different (and vector-space-incompatible!) substitute. The reranker
is treated differently on purpose: if it's missing, search still works
(just without the precision-focused second pass), since search is
read-only against already-committed data with nothing to protect by
failing hard.

## Using it

- **New document** -- write in the editor, hit *Save*. It's a draft:
  freely editable, stored as a small JSON file in your library.
- **Finalize & Save** -- locks the draft forever into a `.sdoc` file in
  your library. This is permanent; there's no "unlock."
- **Download** -- once finalized, grab a portable copy with the
  Download button. It's a completely standard browser download, so
  wherever your browser saves files (or prompts you to choose, if it's
  configured to ask) is where it goes -- Desktop, a synced folder, a USB
  stick, wherever.
- **Search** -- once a document is finalized, search runs a two-stage
  hybrid pipeline (see "How search works" below) blending keyword and
  semantic ranking, then reranking the shortlist for precision.
- **Open a document** -- quick-pick anything already in your library
  from the dropdown, or use the file picker to browse for a `.sdoc` file
  anywhere on disk -- your browser's own native "Open" dialog handles
  the browsing, the file gets uploaded and added to your library, and
  it's immediately searchable.

Everything here is standard browser behavior -- a plain `<input
type=file>` for opening and a plain file download for saving. No native
OS dialogs, no extra permissions, no platform-specific quirks to work
around; it behaves identically in Chrome, Firefox, Safari, and Edge.

## How search works

Search is a two-stage retrieve-then-rerank pipeline, run entirely
in-process against the vectors and FTS5 index already baked into the
`.sdoc` file -- no external vector database, no server, no ANN index
necessary at this scale (a handful to a few hundred chunks per
document; brute-force is both exact and effectively instant here).

**Stage 1 -- recall.** Two independent rankings are computed over every
chunk:
- **Keyword**: SQLite FTS5's BM25, with query terms OR'd together
  (not AND'd -- a natural-language query like "what are the database
  choices" needing *every* word in one chunk would almost never match
  anything; OR lets BM25's own term weighting do the work).
- **Semantic**: cosine similarity between the query's embedding and
  every chunk's stored embedding (brute-force dot product over
  L2-normalized vectors -- exact, not approximate).

These two rankings are combined via **Reciprocal Rank Fusion** (score by
rank position, not raw value) into a candidate shortlist. RRF sidesteps
the fact that BM25 scores and cosine similarities live on completely
incomparable scales -- no normalization scheme reconciles that
reliably, but rank position needs no reconciling at all.

**Stage 2 -- precision.** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
cached the same way as the embedding model) reranks that shortlist by
scoring each `(query, chunk)` pair *jointly* -- more accurate than the
bi-encoder's independently-embedded vectors, because the model actually
reads the query and the chunk together rather than comparing
pre-computed vectors. If the reranker isn't available, search still
works, just ranked by the stage-1 fused score instead -- unlike a
missing embedding model at finalize time, this is never treated as a
hard failure, since search is read-only against data that's already
safely committed.

### Why not a vector database (e.g. Milvus)?

A vector database buys you fast *approximate* search over millions of
vectors, typically via a client-server architecture. Neither applies
here: a single document's chunk count doesn't need approximation to be
fast (brute-force is exact *and* sub-millisecond), and introducing a
server dependency would break the design's core guarantee -- a `.sdoc`
file is a complete, portable, offline-searchable artifact on its own.
If a future "search across my whole library at once" feature ever needs
an index across many documents' worth of chunks, an in-process option
like `sqlite-vec` or FAISS fits this architecture far better than a
server-based database would.

This isn't just theory -- see
[`scripts/experiments/compare_milvus.py`](scripts/experiments/compare_milvus.py)
for a real, measured comparison against Milvus Lite: identical search
rankings, comparable per-query latency, but a ~5.3 **second** one-time
setup cost every time a document's vectors would need loading into a
fresh Milvus collection.

## Project layout

```
app/
  chunking.py    text -> overlapping chunks with source offsets
  embeddings.py  the one local embedding model (sentence-transformers)
  reranker.py    the one local cross-encoder reranker (second-stage search precision)
  sdoc.py        the .sdoc file format: draft, finalize, open, search
  library.py     lists drafts/finalized docs in the library folder
  main.py        FastAPI routes (thin -- logic lives above)
templates/       Jinja2 + HTMX + Tailwind, single-page UI
tests/           pytest coverage (embeddings/reranker are monkeypatched
                 with deterministic test doubles -- see tests/conftest.py
                 -- so the suite runs fast, offline and without needing
                 the real models installed)
```

## What's deferred (by design, for now)

Editing a finalized document, incremental re-embedding, in-place vector
index updates, merge/conflict handling, collaborative editing, and
document sync are all explicitly out of scope for Phase 1. If/when
editing is needed, the plan is to create a new revision file rather than
mutate an existing snapshot -- keeping the core guarantee intact.

## License

MIT -- see [`LICENSE`](LICENSE).
