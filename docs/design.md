# Semantic Document -- Phase 1: Immutable Architecture

Design revision: Phase 1 is immutable after save.

Phase 1 establishes a deliberately strict lifecycle: the user
creates/edits a document in the editor, then performs a Save/Finalize
operation. At that point the document becomes immutable. Its content,
chunks, embeddings, metadata and search indexes are committed together,
as one unit, into a single portable file.

## 1. Phase 1 product contract

- Before Save/Finalize: the document is editable.
- Save/Finalize generates and validates all required embeddings and indexes.
- The save operation succeeds only when content and semantic data are consistent.
- After Save/Finalize: the document is immutable.
- Opening a finalized document never modifies its content or embeddings.
- Search is read-only and operates directly against the embedded semantic data.
- There is exactly one portable user artifact; no external vector database is required.
- If the user wants to change the content, Phase 1 creates a new document/file rather than mutating the old one.

## 2. Revised lifecycle

`Draft -> Validate -> Chunk -> Embed -> Build indexes -> Commit -> LOCKED`

There is no post-save incremental re-embedding workflow in Phase 1.
Incremental editing/index maintenance is deferred to a later phase.

## 3. State machine

- **DRAFT** -- Editable; content can change freely.
- **FINALIZING** -- Editor is locked while chunking, embedding and index construction occur.
- **FINALIZED** -- Immutable, searchable document.
- **FAILED** -- Finalization failed; draft remains available and no finalized artifact is published.

## 4. Atomic finalization

Finalization is a build operation, not an ordinary save. Build the
complete semantic representation first, validate it, and publish the
final single-file artifact only after all components are consistent.

1. Freeze draft content.
2. Canonicalize content.
3. Chunk content and record source offsets.
4. Generate embeddings locally.
5. Build lexical and vector indexes.
6. Write model/version metadata.
7. Run integrity/consistency checks.
8. Commit the finalized artifact atomically.
9. Mark the document immutable.

## 5. Immutable file model

The final `.sdoc` file is a snapshot. It contains the complete
searchable representation required for read-only use:

- Canonical document content
- Document metadata
- Chunk records and source offsets
- Chunk content hashes
- Embeddings
- Embedding-model manifest
- Full-text index
- Vector index
- Format/schema version
- Integrity information
- Optional encryption/security metadata (deferred -- see section 10)

## 6. Why immutability is valuable

- The content and embeddings can never silently drift apart.
- A finalized file is deterministic and reproducible as a semantic snapshot.
- Sharing/copying the file is simpler.
- Concurrent readers do not need to handle mutation.
- Vector indexes never need in-place maintenance after finalization.
- Cryptographic hashing/signing becomes easier.
- Caching and deduplication become easier.
- The file can potentially be content-addressed.
- The security model is simpler because finalized files are read-only.

## 7. Stronger file identity

Because Phase 1 files are immutable, the finalized file can have a
content-derived identity. For example, a canonical document hash can be
calculated over the semantic payload or over a canonicalized logical
representation, e.g.:

```
document_id = hash(canonical content + embedding manifest + semantic metadata)
```

## 8. Search

1. User opens finalized `.sdoc`.
2. Application loads the stored indexes.
3. Query is embedded locally.
4. Vector search retrieves candidate chunks.
5. FTS retrieves exact/keyword matches.
6. Hybrid ranking optionally combines both result sets.
7. The original text is displayed from the immutable document payload.

## 9. Privacy model

Phase 1 assumes the complete finalized artifact is sensitive. Local
embedding generation means document text and queries need not leave the
device.

- No mandatory cloud embedding service.
- No mandatory cloud search service.
- No plaintext sidecar indexes.
- Avoid telemetry containing content, embeddings or search queries.
- (Future) encrypt the complete finalized payload with authenticated
  encryption and a memory-hard password KDF, protecting content,
  embeddings, indexes and metadata together.

## 10. Phase 1 explicitly defers

- Editing a finalized document.
- Incremental re-embedding after save.
- In-place vector-index updates.
- Merge/conflict handling.
- Collaborative editing.
- Document synchronization.
- Multi-version mutable document history.
- Live semantic links that change as content changes.
- Encryption at rest (tracked as a fast-follow, not blocking the MVP).

## 11. Future Phase 2 possibility

If editing a finalized document is later required, the preferred model
should preserve Phase 1's immutability semantics: create a new
revision/file rather than silently mutating the original snapshot. For
example: `research.sdoc` remains immutable; an edit operation produces
`research.v2.sdoc` or a new revision artifact.

## 12. MVP acceptance criteria

- User can create and edit a draft.
- User can finalize/save the draft.
- Finalization generates embeddings before publication.
- Finalization fails safely if embeddings cannot be generated.
- A successful finalized file is immutable.
- Opening a finalized file does not change it.
- Semantic search works without re-indexing.
- The file contains both content and embeddings.
- The file can be copied to another supported machine and searched.
- No separate vector DB or mandatory sidecar index is needed.
- (Deferred) Encrypted mode protects the complete semantic artifact.

## 13. Architectural recommendation

This immutable-first design is materially cleaner for Phase 1 than a
mutable document model. It turns the `.sdoc` into a semantic snapshot
rather than a continuously maintained database. The core abstraction is:

```
CREATE -> FINALIZE -> SEARCH
```

The saved file is the immutable unit of portability, privacy, integrity
and semantic retrieval.
