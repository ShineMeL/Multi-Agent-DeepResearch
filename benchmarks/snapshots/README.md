# Frozen corpus snapshots

Each task snapshot is an immutable directory containing four UTF-8 files:

* `documents.jsonl` — canonical, hash-checked `FrozenEvidenceRecord` rows;
* `index.json` — the deterministic BM25 token rows and index version;
* `snapshot.json` — task identity, versions, count and content hashes; and
* `manifest.sha256` — hashes for the three files above.

Build snapshots offline from private records with:

```text
uv run python -m benchmarks.scripts.build_snapshot one \
  --task-id dev-ts-01 \
  --documents benchmarks/private/frozen_ai_cs_60/documents/dev-ts-01.jsonl \
  --output benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01 \
  --corpus-version ai-cs-60-v1 \
  --index-version bm25-mixed-v1
```

The `batch` subcommand reads private `AnnotatedQuestion` JSONL, resolves one
`<task-id>.jsonl` per question, and processes task IDs lexically. Every child
is self-verified before its staging directory is atomically renamed. Existing
children are never replaced; generated document/index files are ignored by
the repository, while small manifests and public fixtures remain reviewable.
