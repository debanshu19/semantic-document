"""Chunking: split canonical text into overlapping windows, tracking
source offsets so search results can point back to exact spans in the
original document.

Kept deliberately dumb for Phase 1 (MVP) -- paragraph-aware fixed-size
windows with overlap. No sentence-boundary NLP, no tokenizer alignment.
YAGNI until search quality actually demands it.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split `text` into overlapping chunks, preferring paragraph breaks.

    Offsets (start/end) are character positions into the *original*
    canonical `text`, so a chunk can always be traced back to its source.
    """
    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    pos = 0
    length = len(text)
    index = 0

    while pos < length:
        end = min(pos + chunk_size, length)

        # Try to end on a paragraph/sentence boundary near the target end,
        # so we don't slice a word in half if we can help it. Only accept a
        # boundary from the back half of the window -- otherwise a stray
        # paragraph break far from the target end would snap `end` way
        # back, producing a tiny chunk and near-zero forward progress on
        # the next iteration (an earlier version of this code did that).
        if end < length:
            search_start = pos + max(chunk_size // 2, 1)
            boundary = text.rfind("\n\n", search_start, end)
            if boundary == -1:
                boundary = text.rfind(". ", search_start, end)
            if boundary != -1:
                end = boundary + 1

        piece = text[pos:end].strip()
        if piece:
            chunks.append(Chunk(index=index, start=pos, end=end, text=piece))
            index += 1

        if end >= length:
            break
        pos = max(end - overlap, pos + 1)

    return chunks
