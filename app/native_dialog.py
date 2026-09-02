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

macOS focus gotcha: a Tk window created by a plain terminal-launched
Python process (no .app bundle, no Info.plist) doesn't automatically
become the frontmost/focused app the way a normal double-clicked app
would. The dialog genuinely opens -- it's just easy to miss behind your
browser or terminal window. `_bring_to_front()` below fixes that with a
best-effort AppleScript nudge; `-topmost` plus `lift()`/`focus_force()`
handle the rest (and are all that's needed on Windows/Linux).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
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


def _bring_to_front() -> None:
    """Best-effort only: if this fails or isn't permitted (e.g. Automation
    permission not yet granted), we just proceed -- the dialog still
    exists, it just might not be focused. Never let this block/crash the
    actual dialog flow."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to set frontmost of '
                f"(first process whose unix id is {os.getpid()}) to true",
            ],
            timeout=2,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not force this process to the foreground", exc_info=True)


def _hidden_root():
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update()
    _bring_to_front()
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
