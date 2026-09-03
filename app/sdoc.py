"""The .sdoc immutable file format.

A finalized .sdoc is just a SQLite database -- that's the whole trick.
SQLite already gives us: a single portable file, ACID commits, FTS5 for
keyword search, and BLOB storage for embedding vectors. No bespoke binary
format needed, no external vector DB needed.

Lifecycle implemented here (see design doc section 4 "Atomic Finalization"):

    DRAFT (.sdraft.json, freely mutable)
        --finalize()-->
    build in a temp file -> validate -> atomic rename
        -->
    FINALIZED (.sdoc, read-only forever after)

If anything in the build fails, the temp file is discarded and the draft
is left completely untouched. There is no partial/corrupt state a reader
can ever observe.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunking import chunk_text
from app.embeddings import EmbeddingError, embed_query, embed_texts
from app.reranker import rerank as cross_encoder_rerank

SCHEMA_VERSION = 1
DRAFT_SUFFIX = ".sdraft.json"
FINAL_SUFFIX = ".sdoc"


class SDocError(Exception):
    """Any failure in the draft/finalize/open/search lifecycle.

    Per the design doc's FAILED state: finalization failing must never
    corrupt or lose the draft. Every raise site in this module is placed
    *before* any write touches the real .sdoc path.
    """


# --------------------------------------------------------------------------
# Draft management
# --------------------------------------------------------------------------

@dataclass
class Draft:
    title: str
    content: str
    created_at: float
    updated_at: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "content": self.content,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
            indent=2,
        )

    @staticmethod
    def from_json(raw: str) -> "Draft":
        data = json.loads(raw)
        return Draft(**data)


def draft_path(library_dir: Path, name: str) -> Path:
    return library_dir / f"{name}{DRAFT_SUFFIX}"


def final_path(library_dir: Path, name: str) -> Path:
    return library_dir / f"{name}{FINAL_SUFFIX}"


def create_draft(library_dir: Path, name: str, title: str, content: str = "") -> Path:
    library_dir.mkdir(parents=True, exist_ok=True)
    path = draft_path(library_dir, name)
    if path.exists() or final_path(library_dir, name).exists():
        raise SDocError(f"A document named '{name}' already exists.")
    now = time.time()
    path.write_text(Draft(title, content, now, now).to_json(), encoding="utf-8")
    return path


def read_draft(path: Path) -> Draft:
    if not path.exists():
        raise SDocError(f"Draft not found: {path}")
    return Draft.from_json(path.read_text(encoding="utf-8"))


def update_draft(path: Path, title: str, content: str) -> Draft:
    draft = read_draft(path)
    draft.title = title
    draft.content = content
    draft.updated_at = time.time()
    path.write_text(draft.to_json(), encoding="utf-8")
    return draft


# --------------------------------------------------------------------------
# Canonicalization + hashing
# --------------------------------------------------------------------------

def canonicalize(text: str) -> str:
    """Normalize line endings and trailing whitespace so the same logical
    content always hashes identically."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Finalization (build + atomic commit)
# --------------------------------------------------------------------------

def finalize_draft(library_dir: Path, name: str, output_path: Path | None = None) -> Path:
    """Run the full DRAFT -> FINALIZED pipeline for `name`.

    By default the finalized .sdoc is written into `library_dir` next to
    the draft. Pass `output_path` to write it somewhere else entirely --
    e.g. a location the user picked via a native Save dialog. The draft
    itself always lives in `library_dir` regardless of where the
    finalized file ends up.

    Returns the path to the new .sdoc on success. Raises SDocError on any
    failure; the draft is guaranteed untouched in that case.
    """
    d_path = draft_path(library_dir, name)
    f_path = output_path if output_path is not None else final_path(library_dir, name)
    if f_path.exists():
        raise SDocError(f"'{f_path}' already exists and is immutable -- pick a different location.")

    draft = read_draft(d_path)
    canonical = canonicalize(draft.content)
    if not canonical:
        raise SDocError("Cannot finalize an empty document.")

    chunks = chunk_text(canonical)
    if not chunks:
        raise SDocError("Content did not produce any chunks -- nothing to index.")

    try:
        vectors = embed_texts([c.text for c in chunks])
    except EmbeddingError as exc:
        # FAILED state: draft remains available, nothing published.
        raise SDocError(f"Finalization failed while embedding: {exc}") from exc

    if len(vectors) != len(chunks):
        raise SDocError("Embedding count did not match chunk count -- aborting.")

    f_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f_path.parent / f".{f_path.stem}.{uuid.uuid4().hex}.sdoc.tmp"
    try:
        _build_sdoc_file(tmp_path, title=draft.title, canonical=canonical, chunks=chunks, vectors=vectors)
        _verify_integrity(tmp_path, expected_hash=content_hash(canonical), expected_chunks=len(chunks))
        os.replace(tmp_path, f_path)  # atomic on same filesystem
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    d_path.unlink(missing_ok=True)  # content now lives immutably in f_path
    return f_path


def _build_sdoc_file(path: Path, *, title: str, canonical: str, chunks, vectors: np.ndarray) -> None:
    conn = sqlite3.connect(path)
    try:
        # Deliberately NOT WAL mode: WAL leaves -wal/-shm sidecar files
        # sitting next to the database, which would break the "exactly
        # one portable artifact" guarantee (design doc section 1). This
        # is a single write-then-close build, not a long-lived writer, so
        # the default rollback journal (cleaned up automatically on
        # commit) is exactly what we want here.
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE content (id INTEGER PRIMARY KEY CHECK (id = 1), canonical_text TEXT NOT NULL);
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
                vector BLOB NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');
            """
        )
        conn.execute("INSERT INTO content (id, canonical_text) VALUES (1, ?)", (canonical,))

        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                "INSERT INTO chunks (id, start_offset, end_offset, text, content_hash) VALUES (?, ?, ?, ?, ?)",
                (chunk.index, chunk.start, chunk.end, chunk.text, content_hash(chunk.text)),
            )
            conn.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk.index, chunk.text),
            )
            conn.execute(
                "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (chunk.index, vector.astype(np.float32).tobytes()),
            )

        from app.embeddings import EMBEDDING_DIM, MODEL_NAME

        now = time.time()
        meta = {
            "schema_version": str(SCHEMA_VERSION),
            "title": title,
            "status": "FINALIZED",
            "finalized_at": str(now),
            "model_name": MODEL_NAME,
            "embedding_dim": str(EMBEDDING_DIM),
            "content_hash": content_hash(canonical),
            "chunk_count": str(len(chunks)),
        }
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
        conn.commit()
    finally:
        conn.close()


def _verify_integrity(path: Path, *, expected_hash: str, expected_chunks: int) -> None:
    """Read back what we just built and cross-check it before it ever
    becomes visible at the real .sdoc path. Cheap insurance against a
    silently-corrupt build."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT canonical_text FROM content WHERE id = 1").fetchone()
        if row is None or content_hash(row[0]) != expected_hash:
            raise SDocError("Integrity check failed: content hash mismatch.")
        (chunk_count,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        (embed_count,) = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        if chunk_count != expected_chunks or embed_count != expected_chunks:
            raise SDocError("Integrity check failed: chunk/embedding count mismatch.")
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Reading + searching finalized documents (strictly read-only)
# --------------------------------------------------------------------------

@dataclass
class SearchHit:
    chunk_index: int
    text: str
    start: int
    end: int
    score: float
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    rerank_score: float | None = None


def open_finalized(path: Path) -> dict:
    if not path.exists():
        raise SDocError(f"Document not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        (canonical_text,) = conn.execute("SELECT canonical_text FROM content WHERE id = 1").fetchone()
        return {"meta": meta, "content": canonical_text}
    finally:
        conn.close()


CANDIDATE_POOL_SIZE = 20


def search(path: Path, query: str, top_k: int = 5) -> list[SearchHit]:
    """Two-stage hybrid search.

    Stage 1 (recall): keyword (FTS5/BM25) and semantic (cosine
    similarity over stored embeddings) rankings are fused via
    Reciprocal Rank Fusion into a candidate shortlist. RRF combines
    rankings by *position*, not raw score, which matters here because
    BM25 and cosine similarity live on completely incomparable scales --
    no normalization scheme reconciles that reliably, but rank position
    needs no reconciling at all.

    Stage 2 (precision): a cross-encoder reranks that shortlist by
    jointly scoring (query, chunk) pairs -- more accurate than comparing
    independently-embedded vectors, because the model actually sees the
    query and the chunk together. Falls back to the stage-1 fused
    ranking if the reranker isn't available; brute force over a handful
    of chunks is cheap regardless.
    """
    query = query.strip()
    if not query:
        return []

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        chunks = {
            row[0]: {"text": row[1], "start": row[2], "end": row[3]}
            for row in conn.execute("SELECT id, text, start_offset, end_offset FROM chunks")
        }
        if not chunks:
            return []

        raw_keyword = _keyword_scores(conn, query)
        raw_semantic = _semantic_scores(conn, query, list(chunks.keys()))
        keyword_norm = _min_max_normalize(raw_keyword)
        semantic_norm = _min_max_normalize(raw_semantic)

        fused = _reciprocal_rank_fusion(raw_keyword, raw_semantic)
        shortlist_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:CANDIDATE_POOL_SIZE]

        rerank_scores = cross_encoder_rerank(query, [chunks[cid]["text"] for cid in shortlist_ids])

        results: list[SearchHit] = []
        for i, chunk_id in enumerate(shortlist_ids):
            info = chunks[chunk_id]
            rerank_score = rerank_scores[i] if rerank_scores is not None else None
            results.append(
                SearchHit(
                    chunk_index=chunk_id,
                    text=info["text"],
                    start=info["start"],
                    end=info["end"],
                    score=rerank_score if rerank_score is not None else fused[chunk_id],
                    keyword_score=keyword_norm.get(chunk_id, 0.0),
                    semantic_score=semantic_norm.get(chunk_id, 0.0),
                    rerank_score=rerank_score,
                )
            )
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]
    finally:
        conn.close()


def _keyword_scores(conn: sqlite3.Connection, query: str) -> dict[int, float]:
    try:
        rows = conn.execute(
            "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank",
            (_build_fts_query(query),),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS5 raises on malformed MATCH syntax (e.g. bare punctuation) --
        # keyword search is a bonus signal, so degrade gracefully to none.
        return {}
    # bm25() returns *lower is better*; flip sign so higher is better, matching semantic scores.
    return {row[0]: -row[1] for row in rows}


def _build_fts_query(query: str) -> str:
    """Plain space-separated FTS5 MATCH queries default to an implicit
    AND between terms -- so a natural-language query like 'what are the
    database choices' only matches chunks containing *all five* words,
    which is almost never, for almost any real query. OR the terms
    together instead: keyword search becomes 'any of these terms',
    weighted by BM25's own IDF (so common words still contribute little
    relative to rarer, more distinctive ones)."""
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return query
    return " OR ".join(f'"{t}"' for t in terms)


def _semantic_scores(conn: sqlite3.Connection, query: str, chunk_ids: list[int]) -> dict[int, float]:
    query_vector = embed_query(query)
    rows = conn.execute(
        f"SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})",
        chunk_ids,
    ).fetchall()
    scores: dict[int, float] = {}
    for chunk_id, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        # Vectors are pre-normalized at embed time, so dot product == cosine similarity.
        scores[chunk_id] = float(np.dot(query_vector, vector))
    return scores


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _reciprocal_rank_fusion(*rankings: dict[int, float], k: int = 60) -> dict[int, float]:
    """Combine multiple {chunk_id: raw_score} rankings by rank position
    rather than raw score value -- standard, simple, and robust to
    inputs living on totally different scales (BM25 vs. cosine
    similarity aren't comparable numbers). Every chunk gets *some*
    semantic score (cosine similarity is defined for every chunk), so
    the fused result always covers every chunk id."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        ordered = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (chunk_id, _value) in enumerate(ordered, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused
