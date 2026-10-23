# Design notes

Decisions that shaped Retrieva. Prefer this over archaeology in git history.

## Goals

- Self-hostable hybrid retrieval (BM25 + dense) in one binary.
- Incremental upsert/delete with crash recovery.
- Pluggable embedders/rerankers without forcing ONNX on every build.
- Observable defaults: Prometheus always; OTLP behind `otel`.
- Reproducible eval (BEIR-format) for regression floors.

## Non-goals (for now)

- Multi-tenant SaaS control plane.
- Distributed shard rebalancing / consensus.
- Training embedders or ColBERT from scratch inside the repo.
- RocksDB as the default DocStore (filesystem JSON first; RocksDB optional later).

## Why Rust + Tantivy + usearch

- **Rust** — predictable latency, one static-ish binary for Docker, strong typing across traits (`Retriever`, `Embedder`, `QueryCache`).
- **Tantivy** — mature BM25, per-collection directories keep deletes/WAL replay isolated.
- **usearch** — lightweight cosine ANN without a separate vector DB process.

## Hybrid fusion

Default is **RRF** (`rrf_k = 60`): rank-based, scale-free across BM25 and dense scores. Weighted fusion remains available when operators want explicit BM25/dense knobs.

## Reranking

**MaxSim** is primary (hash token embeddings always; ONNX ColBERT optional). Cross-encoder is ablation/compare, not the default path, because it is heavier and needs model files.

## Incremental indexing

WAL-before-apply keeps store and indexes recoverable after crash. Checkpoint is rename-over for durability; Windows requires dropping file handles before rename (learned the hard way).

FsDocStore (JSON files) keeps Phase 6–10 portable on Windows/gnu and Linux without RocksDB toolchain pain. Swap-in RocksDB remains a future feature flag.

## Caching

Layered cache: exact key first, then semantic cosine. Invalidation is collection-scoped on ingest/delete — simple and safe; finer key invalidation can come later if needed.

## Auth and limits

API key is optional shared secret (Bearer or `x-api-key`). Rate limit is process-global token bucket (`governor`), not per-tenant — enough for single-node demos and edge deploys.

## Feature flags

| Feature | Purpose |
|---------|---------|
| (default) | Hash embedder + hash MaxSim; no onnxruntime |
| `onnx` | `ort` embedder + optional ONNX rerankers (rustc ≥ 1.92) |
| `otel` | OTLP span export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set |

## Eval

Smoke fixture under `bench/fixtures/smoke` gates CI with nDCG floor. Full BEIR datasets download via `scripts/download_beir.py` and run offline against the same harness.

## Docker

Image builds `retrieva-cli` with default features (no onnx) for fast pulls. Mount `/app/data` for persistence. Compose profile `metrics` adds Prometheus scraping `/metrics`.
