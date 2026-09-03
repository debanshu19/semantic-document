"""Chunking: split canonical text into overlapping windows, tracking
source offsets so search results can point back to exact spans in the
original document.

Kept deliberately dumb -- no tokenizer alignment, no real NLP sentence
segmentation. Prefers paragraph breaks, then a regex-based sentence
boundary, then a hard cut as a last resort.

Chunk size matters for retrieval quality: too large and each chunk's
embedding averages over multiple unrelated ideas, "diluting" it so it
matches many queries weakly instead of one query strongly. 500 chars
(roughly 2-4 sentences) keeps each chunk's embedding focused on one
idea, which matters more for a small 384-dim bi-encoder than it would
for a larger model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

# A sentence-ending punctuation mark, optionally followed by a closing
# quote/bracket, then whitespace. Not real sentence segmentation (it'll
# occasionally misfire on abbreviations like "Mr. Smith") -- an
# acceptable trade-off for picking a *chunk* boundary, not for anything
# that needs to be linguistically precise.
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def _last_sentence_boundary(text: str, start: int, end: int) -> int:
    """Position right after the last sentence-ending match fully inside
    [start, end), or -1 if none found."""
    last_end = -1
    for m in _SENTENCE_END_RE.finditer(text, start, end):
        last_end = m.end()
    return last_end


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split `text` into overlapping chunks, preferring paragraph breaks,
    then sentence boundaries, over a hard character cut.

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
        # so we don't slice a word (or a thought) in half if we can help
        # it. Only accept a boundary from the back half of the window --
        # otherwise a stray boundary far from the target end would snap
        # `end` way back, producing a tiny chunk and near-zero forward
        # progress on the next iteration.
        if end < length:
            search_start = pos + max(chunk_size // 2, 1)
            boundary = text.rfind("\n\n", search_start, end)
            if boundary != -1:
                end = boundary + 2  # skip both newlines, land cleanly on the next paragraph
            else:
                sentence_end = _last_sentence_boundary(text, search_start, end)
                if sentence_end != -1:
                    end = sentence_end

        piece = text[pos:end].strip()
        if piece:
            chunks.append(Chunk(index=index, start=pos, end=end, text=piece))
            index += 1

        if end >= length:
            break
        pos = max(end - overlap, pos + 1)

    return chunks
