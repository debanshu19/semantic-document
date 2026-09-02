"""FastAPI app: one page that does everything -- open a document, edit
and save a draft, finalize it, and search it. Search results render into
a panel below the editor via HTMX partial swaps (no page reload).

Routes stay thin; all real logic lives in app.sdoc / app.library.

File browsing is handled entirely with standard browser mechanisms --
no OS-specific native dialogs, no extra permissions, no focus-stealing
quirks to work around:
  - Opening a .sdoc from anywhere on disk: a plain `<input type=file>`,
    which the browser renders with the OS's own native file picker and
    uploads the bytes to /open-upload.
  - Saving a finalized .sdoc: it's written into the library, and the
    finalized view offers a `<a download>` link, which the browser's own
    download manager handles (including "ask where to save" if the
    user's browser is configured that way).

Plain web app, run with `uvicorn app.main:app`. Document data lives in
the user's home directory rather than next to the source code, so the
*documents* stay portable (copy `~/SemanticDocument/library/` anywhere)
even though the app itself is just a regular local web server.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import sdoc
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
def finalize(doc: str = Form(...)):
    try:
        sdoc.finalize_draft(LIBRARY_DIR, doc)
    except sdoc.SDocError as exc:
        return RedirectResponse(f"/?doc={quote(doc)}&error={quote(str(exc))}", status_code=303)
    return RedirectResponse(f"/?doc={quote(doc)}", status_code=303)


@app.get("/documents/{name}/download")
def download(name: str):
    """Serves a finalized .sdoc for the browser's own download manager to
    handle -- this is the 'ask the user where to save it' step, done
    entirely with a standard HTTP download rather than any OS-specific
    dialog."""
    final = sdoc.final_path(LIBRARY_DIR, name)
    if not final.exists():
        raise HTTPException(404, "Document not found.")
    return FileResponse(final, filename=final.name, media_type="application/octet-stream")


@app.post("/open-upload")
async def open_upload(request: Request, file: UploadFile):
    """Opening a .sdoc 'from anywhere on disk' via a plain browser file
    input -- the browser renders the OS's native file picker itself, so
    there's no server-side dialog/focus trickery needed at all. The
    uploaded bytes are validated and copied into the library so the
    document becomes a normal, searchable library entry from then on.
    """
    raw_name = Path(file.filename or "imported").stem
    base_slug = _slugify(raw_name)
    content = await file.read()

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=LIBRARY_DIR, suffix=".sdoc.uploadtmp", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        sdoc.open_finalized(tmp_path)  # validates it's actually a well-formed .sdoc
    except Exception:  # noqa: BLE001 - any failure means "not a valid .sdoc"
        tmp_path.unlink(missing_ok=True)
        entries = list_library(LIBRARY_DIR)
        return _render(request, entries, mode="new", error=f"'{file.filename}' is not a valid .sdoc file.")

    name = _unique_name(base_slug)
    os.replace(tmp_path, sdoc.final_path(LIBRARY_DIR, name))
    return RedirectResponse(f"/?doc={quote(name)}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = "", doc: str = "", path: str = ""):
    target = Path(unquote(path)).expanduser() if path else sdoc.final_path(LIBRARY_DIR, doc)
    if not target.exists():
        raise HTTPException(404, "Document not found.")
    hits = sdoc.search(target, q) if q.strip() else []
    return templates.TemplateResponse(request, "_search_results.html", {"hits": hits, "query": q})
