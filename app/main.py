"""FastAPI app: one page that does everything -- open a document, edit
and save a draft, finalize it, and search it. Search results render into
a panel below the editor via HTMX partial swaps (no page reload).

Routes stay thin; all real logic lives in app.sdoc / app.library.

Plain web app, run with `uvicorn app.main:app`. Document data lives in
the user's home directory rather than next to the source code, so the
*documents* stay portable (copy `~/SemanticDocument/library/` anywhere)
even though the app itself is just a regular local web server.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import native_dialog, sdoc
from app.library import list_library

BASE_DIR = Path(__file__).resolve().parent.parent
LIBRARY_DIR = Path(os.environ.get("SDOC_LIBRARY_DIR", Path.home() / "SemanticDocument" / "library"))
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="Semantic Document")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip()).strip("-").lower()
    return slug or "untitled"


def _unique_name(base_slug: str) -> str:
    name = base_slug
    suffix = 2
    while sdoc.draft_path(LIBRARY_DIR, name).exists() or sdoc.final_path(LIBRARY_DIR, name).exists():
        name = f"{base_slug}-{suffix}"
        suffix += 1
    return name


def _render(request, entries, *, mode, doc_ref=None, is_external=False, draft=None, finalized=None, error=""):
    return templates.TemplateResponse(
        request,
        "app.html",
        {
            "entries": entries,
            "mode": mode,  # "new" | "draft" | "finalized"
            "doc_ref": doc_ref,
            "is_external": is_external,
            "draft": draft,
            "finalized": finalized,
            "error": error,
            "dialogs_available": native_dialog.available(),
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request, doc: str = "", path: str = "", error: str = ""):
    """The one page. `doc` opens a library entry by name, `path` opens any
    finalized .sdoc file elsewhere on disk. Neither given -> blank new draft."""
    entries = list_library(LIBRARY_DIR)

    if path:
        external = Path(unquote(path)).expanduser()
        if not external.exists() or external.suffix != sdoc.FINAL_SUFFIX:
            return _render(request, entries, mode="new", error=f"Not a valid .sdoc file: {external}")
        try:
            opened = sdoc.open_finalized(external)
        except sdoc.SDocError as exc:
            return _render(request, entries, mode="new", error=str(exc))
        return _render(
            request, entries, mode="finalized", doc_ref=str(external), is_external=True,
            finalized=opened, error=error,
        )

    if doc:
        final = sdoc.final_path(LIBRARY_DIR, doc)
        if final.exists():
            opened = sdoc.open_finalized(final)
            return _render(
                request, entries, mode="finalized", doc_ref=doc, is_external=False,
                finalized=opened, error=error,
            )
        draft_p = sdoc.draft_path(LIBRARY_DIR, doc)
        if draft_p.exists():
            draft = sdoc.read_draft(draft_p)
            return _render(request, entries, mode="draft", doc_ref=doc, draft=draft, error=error)
        return _render(request, entries, mode="new", error=f"Document '{doc}' not found.")

    return _render(request, entries, mode="new", error=error)


@app.post("/save")
def save(title: str = Form(...), content: str = Form(...), doc: str = Form("")):
    if doc:
        sdoc.update_draft(sdoc.draft_path(LIBRARY_DIR, doc), title=title, content=content)
        name = doc
    else:
        name = _unique_name(_slugify(title))
        sdoc.create_draft(LIBRARY_DIR, name, title=title, content=content)
    return RedirectResponse(f"/?doc={quote(name)}", status_code=303)


@app.post("/finalize")
async def finalize(doc: str = Form(...)):
    """Finalizing is the one truly permanent 'save' in this app -- so it's
    the one place we pop a native Save dialog and ask the user exactly
    where the resulting .sdoc should live. This must stay `async def`
    (see app/native_dialog.py) so the dialog runs on the same thread as
    the event loop rather than a thread-pool worker.
    """
    chosen = native_dialog.ask_save_path(default_name=doc)
    if chosen is None:
        if native_dialog.available():
            # User hit Cancel in the dialog -- not an error, just a no-op.
            return RedirectResponse(f"/?doc={quote(doc)}", status_code=303)
        # No display/Tk available at all (e.g. headless server) -- fall
        # back to the old behavior of finalizing straight into the library.
        chosen = sdoc.final_path(LIBRARY_DIR, doc)

    try:
        sdoc.finalize_draft(LIBRARY_DIR, doc, output_path=chosen)
    except sdoc.SDocError as exc:
        return RedirectResponse(f"/?doc={quote(doc)}&error={quote(str(exc))}", status_code=303)
    return RedirectResponse(f"/?path={quote(str(chosen))}", status_code=303)


@app.get("/browse/open")
async def browse_open(doc: str = ""):
    """Pops a native 'Open' dialog filtered to .sdoc files. `doc` is just
    carried through so Cancel returns you to wherever you were."""
    chosen = native_dialog.ask_open_path()
    if chosen is None:
        fallback = f"/?doc={quote(doc)}" if doc else "/"
        return RedirectResponse(fallback, status_code=303)
    return RedirectResponse(f"/?path={quote(str(chosen))}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = "", doc: str = "", path: str = ""):
    target = Path(unquote(path)).expanduser() if path else sdoc.final_path(LIBRARY_DIR, doc)
    if not target.exists():
        raise HTTPException(404, "Document not found.")
    hits = sdoc.search(target, q) if q.strip() else []
    return templates.TemplateResponse(request, "_search_results.html", {"hits": hits, "query": q})
