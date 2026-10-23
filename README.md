# Retrieva

Self-hostable **hybrid retrieval engine** in Rust for RAG and document search.

Ingest local files, index them with **BM25 + dense vectors**, fuse results, optionally rerank, and serve search over **REST / gRPC**. Built for private corpora—you keep the data on your machine.

```text
docs / PDFs  →  ingest  →  Tantivy BM25 + usearch ANN  →  fuse  →  cache  →  HTTP/gRPC
```

## Features

- **Hybrid search** — BM25 (Tantivy), dense ANN (usearch), RRF or weighted fusion
- **Embeddings** — fast hash embedder for demos; real **BGE-small** via ONNX (`--features onnx`)
- **Reranking** — ColBERT-style MaxSim and cross-encoder (quality path; CPU latency is high)
- **Query cache** — exact + semantic, with shared query embeddings and hit-rate reporting
- **Durable indexes** — WAL + filesystem store; survives restarts
- **Ops** — health, Prometheus metrics, optional API key + rate limits
- **Benchmarks** — BEIR SciFact matrix, Zipf load tests, CE vs ColBERT ablate

## Quick start (Docker)

```bash
docker compose up --build -d
curl -s http://localhost:8080/health

docker compose exec retrieva retrieva ingest \
  --collection demo /app/demo/docs --embedder hash

curl -s http://localhost:8080/v1/search \
  -H 'content-type: application/json' \
  -d '{"collection_id":"demo","query":"hybrid BM25 dense retrieval","top_k":5,"mode":"hybrid","use_cache":true}'
```

Full walkthrough (auth, metrics, teardown): [docs/DEMO.md](docs/DEMO.md).

## Quick start (CLI)

```bash
cargo build -p retrieva-cli --release

cargo run -p retrieva-cli -- ingest --collection demo ./demo/docs --embedder hash
cargo run -p retrieva-cli -- query --collection demo "hybrid retrieval" --mode hybrid
cargo run -p retrieva-cli -- serve   # REST :8080, gRPC :50051
```

Config: [config/retrieva.example.toml](config/retrieva.example.toml) (override with `RETRIEVA_*` env).

**ONNX / Windows:** prefer the `retrieva-onnx:builder` Docker image for `--features onnx`. Native Windows needs a working MSVC/CUDA toolchain for ORT.

```bash
python scripts/download_bge_small.py
cargo run -p retrieva-cli --features onnx --release -- \
  ingest --collection demo ./demo/docs --embedder onnx
```

Models and paths: [docs/MODELS.md](docs/MODELS.md).

## Search API

| Surface | Endpoints |
|---------|-----------|
| REST | `GET /health`, `GET /metrics`, `POST /v1/search`, `POST /v1/ingest` |
| gRPC | `retrieva.v1.Retrieva` — Health, Search, Ingest |

Modes: `bm25` · `dense` · `hybrid` · `hybrid+rerank` (`use_cache` optional).

## Benchmarks (SciFact, ONNX)

Hardware: Docker Linux x86_64, 16 CPUs, CPU-only ORT unless noted. Reproduce from [bench/results/](bench/results/).

### Retrieval quality (`bge-small-en-v1.5`)

| Mode | nDCG@10 | Recall@10 | MRR |
|------|--------:|----------:|----:|
| BM25 | 0.6612 | 0.7876 | 0.6276 |
| Dense | **0.6845** | 0.7980 | 0.6550 |
| Hybrid | 0.6673 | 0.7983 | 0.6321 |

On SciFact, **dense wins**; hybrid is close but does not beat the best single by ≥0.01 (`hybrid_ok=false`). Hash-embedder matrices are inconclusive and not cited.

### Reranking (same queries; CPU)

| Candidates | Backend | nDCG@10 | Latency |
|------------|---------|--------:|--------:|
| dense | (no rerank) | **0.6845** | — |
| hybrid | (no rerank) | 0.6673 | — |
| hybrid | ColBERT MaxSim | 0.7020 | ~3.3 s / query @ top-50 |
| hybrid | Cross-encoder | 0.7016 | ~15 s / query @ top-50 |

ColBERT is about **+0.018** over dense alone—not a large jump—and these latencies are **not interactive** on CPU (50 candidates; docs were re-encoded per query in that run). Doc-token caching, smaller pools, and `--features cuda` are the path to faster rerank.

### Throughput (ONNX hybrid, Zipf queries, 100 req @ c=4)

| Cache | p50 | Hit rate | QPS |
|-------|----:|---------:|----:|
| off | 140 ms | — | **25.7** |
| on | 107 ms | 42% | **52.3** |

With a realistic Zipf mix, cache helps. Unique-only traffic + a double embed used to make cache-on *slower*; that path is fixed.

Also: embed parity vs Python `1.000000` — [bench/results/embed_parity_batch.md](bench/results/embed_parity_batch.md).

```bash
python scripts/download_beir.py scifact
python scripts/download_bge_small.py   # + reranker / ColBERT as needed
# then: retrieva bench --suite matrix|load|rerank-ablate --embedder onnx
```

## Project layout

| Path | Role |
|------|------|
| `crates/*` | core, ingest, index, fusion, rerank, cache, server, bench, cli |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [DESIGN](docs/DESIGN.md), [STORAGE](docs/STORAGE.md), [DEMO](docs/DEMO.md) |
| `demo/` | Small sample docs + arXiv-style smoke corpus |
| `bench/` | Fixtures + checked-in result tables |
| `proto/` | gRPC definitions |

## Stack notes

| Area | Choice |
|------|--------|
| License | Apache-2.0 |
| Rust | Edition 2021; MSRV 1.90 (default). ONNX needs rustc ≥ 1.92 (`ort`) |
| Sparse / dense | Tantivy BM25 · usearch cosine |
| ONNX | `ort = 2.0.0-rc.15`, optional CUDA EP |
| CI | `ubuntu-latest`, default features (no ONNX) — `make ci` |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
