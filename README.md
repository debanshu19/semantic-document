# Semantic Document

Write it. Finalize it. It's locked forever, and now it's semantically
searchable -- offline, with no external vector database required.

## The idea

Most "AI search over my notes" tools keep your text and its embeddings in
two separate places that can silently drift apart. Semantic Document
takes a different, deliberately strict approach for its first phase:

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
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

Your documents live in `~/SemanticDocument/library/` -- not inside the
repo. That keeps them independent of wherever you happen to have checked
out the source, and safe from `git clean`. Override the location with
the `SDOC_LIBRARY_DIR` environment variable if you want it elsewhere.

The first finalize will try to download a small local embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~80MB) from Hugging Face --
install it yourself first with `uv pip install sentence-transformers` if
you want that. Without it, the app automatically uses a lightweight
built-in hashing embedder instead (see "Embedding backends" below), so
it works fully offline with zero extra setup either way.

## Using it

- **New document** -- write in the editor, hit *Save*. It's a draft:
  freely editable, stored as a small JSON file in your library.
- **Finalize & Save** -- locks the draft forever into a `.sdoc` file.
  This is permanent; there's no "unlock."
- **Search** -- once a document is finalized, the search box blends
  keyword (FTS5/BM25) and semantic (cosine similarity) ranking over the
  chunks stored inside the file.
- **Open a document** -- pick any document from your library, or open
  *any* `.sdoc` file elsewhere on disk by path -- it's fully portable.

## Embedding backends

`app/embeddings.py` picks a backend automatically, once per process:

- **sentence-transformers** (if installed and loadable) -- the real
  local transformer model, best quality.
- **hashing** (always available, zero dependencies) -- a deterministic
  feature-hashed bag-of-words embedder. Lower semantic quality, but a
  perfectly honest fallback that needs nothing beyond numpy and works
  identically across machines and process restarts.

Whichever one actually produced a document's embeddings gets recorded in
that document's own metadata, so `.sdoc` files stay self-describing no
matter which backend built them.

## Project layout

```
app/
  chunking.py    text -> overlapping chunks with source offsets
  embeddings.py  local embedding backends (sentence-transformers + hashing fallback)
  sdoc.py        the .sdoc file format: draft, finalize, open, search
  library.py     lists drafts/finalized docs in the library folder
  main.py        FastAPI routes (thin -- logic lives above)
templates/       Jinja2 + HTMX + Tailwind, single-page UI
tests/           pytest coverage for the core lifecycle
```

## What's deferred (by design, for now)

Editing a finalized document, incremental re-embedding, in-place vector
index updates, merge/conflict handling, collaborative editing, and
document sync are all explicitly out of scope for Phase 1. If/when
editing is needed, the plan is to create a new revision file rather than
mutate an existing snapshot -- keeping the core guarantee intact.

## License

MIT -- see [`LICENSE`](LICENSE).
