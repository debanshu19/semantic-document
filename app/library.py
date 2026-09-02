"""Lists documents living in the library folder -- pairs up each
basename with its current lifecycle state (DRAFT or FINALIZED).

This is intentionally the *only* place that scans the filesystem, so the
web layer never has to know about .sdraft.json / .sdoc naming details.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.sdoc import DRAFT_SUFFIX, FINAL_SUFFIX, read_draft, open_finalized


@dataclass
class LibraryEntry:
    name: str
    title: str
    status: str  # "DRAFT" | "FINALIZED"
    updated_at: float | None = None


def list_library(library_dir: Path) -> list[LibraryEntry]:
    library_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, LibraryEntry] = {}

    for path in sorted(library_dir.glob(f"*{FINAL_SUFFIX}")):
        name = path.name[: -len(FINAL_SUFFIX)]
        doc = open_finalized(path)
        entries[name] = LibraryEntry(
            name=name,
            title=doc["meta"].get("title", name),
            status="FINALIZED",
            updated_at=float(doc["meta"].get("finalized_at", 0)),
        )

    for path in sorted(library_dir.glob(f"*{DRAFT_SUFFIX}")):
        name = path.name[: -len(DRAFT_SUFFIX)]
        if name in entries:
            continue  # a finalized doc always wins over a stray leftover draft
        draft = read_draft(path)
        entries[name] = LibraryEntry(
            name=name,
            title=draft.title,
            status="DRAFT",
            updated_at=draft.updated_at,
        )

    return sorted(entries.values(), key=lambda e: e.updated_at or 0, reverse=True)
