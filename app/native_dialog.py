"""Native OS file dialogs for choosing where documents get saved to and
opened from.

This app runs entirely on localhost for a single local user -- server
and browser are always the same machine -- so instead of building a
custom in-browser directory browser (or relying on the File System
Access API, which only Chromium browsers support), we just pop the OS's
own native "Save As" / "Open" dialog. tkinter ships with Python and
gives us this for free on macOS, Windows and Linux alike.

Threading note: on macOS in particular, Tk/Cocoa UI must run on the
thread the interpreter started on. A plain `uvicorn app.main:app`
process runs its asyncio event loop on the main thread, and Starlette
calls `async def` route handlers directly on that loop (no thread-pool
hop) -- so calling these functions from an `async def` route, without
wrapping them in run_in_threadpool/asyncio.to_thread, is what keeps
this main-thread-safe. It also blocks the server while the dialog is
open, which is fine (even correct) for a single-user local tool.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.sdoc import FINAL_SUFFIX

logger = logging.getLogger(__name__)

_FILETYPES = [("Semantic Document", f"*{FINAL_SUFFIX}"), ("All files", "*.*")]


def available() -> bool:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _hidden_root():
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def ask_save_path(default_name: str) -> Path | None:
    """Native 'Save As' dialog. Returns the chosen path, or None if the
    user cancelled (or no display/Tk is available -- callers should fall
    back to a sensible default in that case)."""
    if not available():
        return None
    from tkinter import filedialog

    root = _hidden_root()
    try:
        chosen = filedialog.asksaveasfilename(
            parent=root,
            title="Save finalized document as",
            defaultextension=FINAL_SUFFIX,
            initialfile=f"{default_name}{FINAL_SUFFIX}",
            filetypes=_FILETYPES,
        )
    except Exception:  # noqa: BLE001 - headless/no-display environments etc.
        logger.warning("Native save dialog unavailable", exc_info=True)
        return None
    finally:
        root.destroy()
    return Path(chosen) if chosen else None


def ask_open_path() -> Path | None:
    """Native 'Open' dialog filtered to .sdoc files. Returns the chosen
    path, or None if the user cancelled (or no display/Tk is available)."""
    if not available():
        return None
    from tkinter import filedialog

    root = _hidden_root()
    try:
        chosen = filedialog.askopenfilename(
            parent=root,
            title="Open a Semantic Document",
            filetypes=_FILETYPES,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Native open dialog unavailable", exc_info=True)
        return None
    finally:
        root.destroy()
    return Path(chosen) if chosen else None
