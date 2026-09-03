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
./scripts/install-embeddings.sh   # the embedding model -- see below
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

Your documents live in `~/SemanticDocument/library/` -- not inside the
repo. That keeps them independent of wherever you happen to have checked
out the source, and safe from `git clean`. Override the location with
the `SDOC_LIBRARY_DIR` environment variable if you want it elsewhere.

Semantic Document uses exactly one embedding model --
`sentence-transformers/all-MiniLM-L6-v2` -- running fully locally, so no
document text or query ever leaves your machine. It's kept as an
optional (rather than hard) dependency only because it pulls in torch, a
large download; functionally, it's the app's one and only embedding
backend, no fallback.

`scripts/install-embeddings.sh` handles the one-time setup: installs the
package and pre-downloads/caches the model weights (~80MB, cached under
`~/.cache/huggingface`). After that, the app runs the model in Hugging
Face's **offline mode** by default (`app/embeddings.py` sets
`HF_HUB_OFFLINE=1`) -- it will never make a network call on its own.
That's not just a corporate-network workaround; it's the correct
behavior for an app whose whole privacy model is "no cloud calls, ever."
If the model isn't cached yet, that's a setup step to run once, not
something finalize should silently reach out to the internet for.

If it's not installed/cached, finalize fails safely and clearly: your
draft is untouched, nothing half-finished gets published. That's the
design doc's intended FAILED state working as designed, not a bug --
there's no silent degrading to a lower-quality substitute model.

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
- **Search** -- once a document is finalized, the search box blends
  keyword (FTS5/BM25) and semantic (cosine similarity) ranking over the
  chunks stored inside the file.
- **Open a document** -- quick-pick anything already in your library
  from the dropdown, or use the file picker to browse for a `.sdoc` file
  anywhere on disk -- your browser's own native "Open" dialog handles
  the browsing, the file gets uploaded and added to your library, and
  it's immediately searchable.

Everything here is standard browser behavior -- a plain `<input
type=file>` for opening and a plain file download for saving. No native
OS dialogs, no extra permissions, no platform-specific quirks to work
around; it behaves identically in Chrome, Firefox, Safari, and Edge.

## Project layout

```
app/
  chunking.py    text -> overlapping chunks with source offsets
  embeddings.py  the one local embedding model (sentence-transformers)
  sdoc.py        the .sdoc file format: draft, finalize, open, search
  library.py     lists drafts/finalized docs in the library folder
  main.py        FastAPI routes (thin -- logic lives above)
templates/       Jinja2 + HTMX + Tailwind, single-page UI
tests/           pytest coverage (embeddings are monkeypatched with a
                 deterministic test double -- see tests/conftest.py --
                 so the suite runs fast, offline and without needing
                 the real model installed)
```

## What's deferred (by design, for now)

Editing a finalized document, incremental re-embedding, in-place vector
index updates, merge/conflict handling, collaborative editing, and
document sync are all explicitly out of scope for Phase 1. If/when
editing is needed, the plan is to create a new revision file rather than
mutate an existing snapshot -- keeping the core guarantee intact.

## License

MIT -- see [`LICENSE`](LICENSE).
