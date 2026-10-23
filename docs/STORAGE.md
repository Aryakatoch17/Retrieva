# Storage model (honest)

Retrieva persists under `data_dir` with three cooperating layers:

| Layer | Path | Role | Durability |
|-------|------|------|------------|
| **WAL** | `data_dir/wal/log.jsonl` + `checkpoint.json` | Ordered upsert/delete ops before apply | Append + fsync; checkpoint via temp+rename |
| **FsDocStore** | `data_dir/store/` | Collection / document / chunk JSON metadata | Per-file write+rename (not a transactional DB) |
| **Indexes** | `data_dir/indexes/<col>/tantivy`, `…/usearch` | BM25 + ANN | Library-native persistence after each batch |

## What is implemented

- **WAL crash recovery (tested):** if the process dies after `append` but before `checkpoint`, the next `IncrementalIndex::open` replays pending ops into store + Tantivy + usearch. Covered by `incremental::tests::crash_recovery_replays_wal`.
- **FsDocStore:** portable JSON files. Fine for demos, single-node labs, and moderate corpora. **Not** RocksDB — no multi-process writers, no LSM compaction, no secondary indexes beyond `doc_chunks` sidecars.
- **Deletes:** document delete removes store rows and sparse/dense postings; WAL records the op.

## What is *not* claimed

- Cross-process concurrent writers to the same `data_dir`.
- Point-in-time backup consistency without quiescing writers (copy `data_dir` only when idle or after checkpoint).
- Drop-in RocksDB (deferred; Windows/gnu tooling made it a poor default).

## Operator guidance

1. Prefer one writer process (`serve` or CLI ingest) per `data_dir`.
2. After unclean shutdown, restart is enough — recovery runs on open.
3. For production at scale, plan a future RocksDB (or SQLite) DocStore behind the same `DocStore` trait; WAL semantics stay.
