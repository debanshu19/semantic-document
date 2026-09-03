"""EXPERIMENTAL comparison script -- not part of the app.

Compares our current approach (brute-force numpy cosine similarity over
vectors stored in the .sdoc file) against Milvus Lite (embedded,
file-based ANN search) for a single document's worth of chunks.

Uses identical chunking (app.chunking) and identical embeddings
(app.embeddings, real sentence-transformers model) for both approaches
-- Milvus does not generate embeddings itself, so the only variable
under test here is the vector search backend.

Run with: uv run python scripts/experiments/compare_milvus.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from app.chunking import chunk_text
from app.embeddings import embed_query, embed_texts

DOCUMENT = """
Chapter 8: Not All Storage Should Be the Same.

A common mistake in system design is asking: which database should we use?
For a system like real-time location tracking, the better question is:
what type of data are we storing, and how will we access it?

You may need multiple storage layers. Hot data, like a driver's latest
location, needs extremely fast reads and writes, so a memory-oriented or
highly optimized real-time store may be appropriate. Durable data, such
as trip-started or payment-completed events, requires stronger durability
guarantees and often lands in a write-ahead log or an append-only store.
Historical data used for analytics, demand prediction, and fraud
detection may use a completely different storage system entirely, often
a columnar warehouse optimized for large scans rather than point lookups.

Chapter 9: The Hidden Enemy: Hotspots.

Not every geographic area receives equal traffic. Consider a remote
village with twenty drivers versus an airport with fifty thousand
drivers. If every driver near a busy airport sends location updates
simultaneously, a naive partitioning scheme can create a hotspot that
overwhelms a single shard while the rest of the cluster sits idle.
A common fix is to salt the partition key with a random suffix, spreading
writes for a single hot entity across multiple shards, then fanning the
read back out across all of them at query time.

Chapter 10: Choosing a Consistency Model.

Strong consistency guarantees that every read sees the latest write, but
it usually costs latency and availability during network partitions.
Eventual consistency trades that guarantee for lower latency and higher
availability, accepting that reads may briefly return stale data.
For a ride-hailing location system, eventual consistency is often the
right trade-off: a driver's position being a second or two stale rarely
matters, but the system staying available during a network blip does.

Chapter 11: Caching Strategies.

A read-through cache sits in front of the database and is populated
lazily on cache misses. A write-through cache is updated synchronously
whenever the underlying data changes, keeping the cache and the source
of truth consistent at the cost of extra write latency. Cache invalidation
remains one of the two genuinely hard problems in computer science, and
choosing the wrong TTL can either serve stale data for too long or
thrash the origin database with redundant reads.

Chapter 12: API Design for High-Throughput Systems.

Pagination, idempotency keys, and rate limiting are not optional
extras for a high-throughput API -- they are load-bearing walls.
Without idempotency keys, a retried request after a timeout can create
duplicate side effects, like charging a customer twice. Rate limiting
protects the system from both malicious abuse and well-intentioned but
poorly-behaved clients that retry too aggressively during an outage.
""".strip()

QUERIES = [
    "what are the database choices",
    "how do you handle hotspots and uneven traffic",
    "difference between strong and eventual consistency",
    "how does caching work",
    "why do we need idempotency keys",
]


def run_numpy_search(chunks_text: list[str], vectors: np.ndarray, query: str, top_k: int = 3):
    start = time.perf_counter()
    query_vec = embed_query(query)
    scores = vectors @ query_vec  # vectors pre-normalized -> dot product == cosine similarity
    order = np.argsort(-scores)[:top_k]
    elapsed = time.perf_counter() - start
    return [(int(i), float(scores[i]), chunks_text[i]) for i in order], elapsed


def main():
    print("=" * 70)
    print(" Building chunks + embeddings (identical for both approaches)")
    print("=" * 70)
    chunks = chunk_text(DOCUMENT)
    texts = [c.text for c in chunks]
    print(f"{len(texts)} chunks produced")
    vectors = embed_texts(texts)
    print(f"Embedded to shape {vectors.shape}")

    # ---- Our current approach: brute-force numpy -----------------------
    print()
    print("=" * 70)
    print(" Approach A: brute-force numpy cosine similarity (current)")
    print("=" * 70)
    numpy_results = {}
    numpy_times = []
    for q in QUERIES:
        results, elapsed = run_numpy_search(texts, vectors, q)
        numpy_results[q] = results
        numpy_times.append(elapsed)
        print(f"\nQuery: {q!r}  ({elapsed * 1000:.3f}ms)")
        for rank, (idx, score, text) in enumerate(results, 1):
            print(f"  #{rank} chunk {idx} score={score:.4f}: {text[:80]}...")

    # ---- Milvus Lite: embedded, file-based ANN search -------------------
    print()
    print("=" * 70)
    print(" Approach B: Milvus Lite (embedded ANN search)")
    print("=" * 70)

    db_path = Path("/tmp/milvus_experiment.db")
    db_path.unlink(missing_ok=True)
    for suffix in (".db-wal", ".db-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)

    setup_start = time.perf_counter()
    from pymilvus import MilvusClient

    client = MilvusClient(str(db_path))
    client.create_collection(collection_name="doc_chunks", dimension=vectors.shape[1], metric_type="COSINE")
    client.insert(
        collection_name="doc_chunks",
        data=[{"id": i, "vector": vectors[i].tolist(), "text": texts[i]} for i in range(len(texts))],
    )
    setup_elapsed = time.perf_counter() - setup_start
    print(f"\nCollection create + insert (this is the 'load onto Milvus on every doc open' cost): {setup_elapsed * 1000:.2f}ms")

    milvus_results = {}
    milvus_times = []
    for q in QUERIES:
        start = time.perf_counter()
        query_vec = embed_query(q)
        hits = client.search(
            collection_name="doc_chunks",
            data=[query_vec.tolist()],
            limit=3,
            output_fields=["text"],
        )[0]
        elapsed = time.perf_counter() - start
        milvus_results[q] = [(h["id"], h["distance"], h["entity"]["text"]) for h in hits]
        milvus_times.append(elapsed)
        print(f"\nQuery: {q!r}  ({elapsed * 1000:.3f}ms)")
        for rank, (idx, score, text) in enumerate(milvus_results[q], 1):
            print(f"  #{rank} chunk {idx} score={score:.4f}: {text[:80]}...")

    client.close()
    for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass  # Milvus Lite can hold a lock briefly after close(); harmless for a one-off script

    # ---- Comparison -------------------------------------------------------
    print()
    print("=" * 70)
    print(" COMPARISON")
    print("=" * 70)

    print(f"\nDocument: {len(texts)} chunks, {vectors.shape[1]}-dim vectors")
    print(f"\nAvg query latency (search only, no setup):")
    print(f"  numpy brute force : {sum(numpy_times) / len(numpy_times) * 1000:.3f}ms")
    print(f"  Milvus Lite       : {sum(milvus_times) / len(milvus_times) * 1000:.3f}ms")
    print(f"\nMilvus one-time setup cost per document open (create collection + insert + index):")
    print(f"  {setup_elapsed * 1000:.2f}ms  <-- this happens EVERY time a doc is opened, per the proposed flow")

    print(f"\nRanking agreement (top-1 chunk match per query):")
    agree = 0
    for q in QUERIES:
        numpy_top1 = numpy_results[q][0][0]
        milvus_top1 = milvus_results[q][0][0]
        match = "MATCH" if numpy_top1 == milvus_top1 else "DIFFERENT"
        if numpy_top1 == milvus_top1:
            agree += 1
        print(f"  {q!r}: numpy=chunk{numpy_top1} milvus=chunk{milvus_top1} -- {match}")
    print(f"\n{agree}/{len(QUERIES)} queries agree on the top-1 result")


if __name__ == "__main__":
    main()
