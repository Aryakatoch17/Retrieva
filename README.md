# Retrieva

Self-hostable **hybrid RAG retrieval engine** in Rust: BM25 + dense ANN, late-interaction / cross-encoder reranking, incremental indexing, semantic cache, and a reproducible BEIR benchmark suite.

> **Status:** Phase 10 complete — Docker demo, architecture docs, design notes.

## Benchmarks (ONNX / real models — SciFact)

Hardware unless noted: Docker `retrieva-onnx:builder`, Linux x86_64, **16 logical CPUs**, CPU-only ORT (CUDA EP available via `--features cuda` when the host has a compatible toolkit).

### Mode matrix (`bge-small-en-v1.5`)

[`matrix_onnx_scifact.json`](bench/results/matrix_onnx_scifact.json) — `hybrid_ok=false` (**MIXED**: hybrid does not clear +0.01 over best single; dense wins here).

| Mode | nDCG@10 | Recall@10 | MRR |
|------|--------:|----------:|----:|
| Bm25 | 0.6612 | 0.7876 | 0.6276 |
| Dense | **0.6845** | 0.7980 | 0.6550 |
| Hybrid | 0.6673 | 0.7983 | 0.6321 |

`HybridRerank` is **not** in this matrix. Hash MaxSim (~0.39 nDCG) is a smoke test only and must not be cited as a result. Real CE / ColBERT rows live in the ablate table below.

Hash-embedder matrices are inconclusive for hybrid and are not cited here.

### CE vs ColBERT MaxSim (with dense baseline)

Same SciFact queries — first-stage candidates then optional rerank. Prior CPU numbers (candidates re-encoded every query; **not interactive**):

| Candidates | Backend | nDCG@10 | mean latency |
|------------|---------|--------:|-------------:|
| hybrid | identity (no rerank) | 0.6673 | — |
| dense | identity (no rerank) | **0.6845** | — |
| hybrid | ColBERT MaxSim | 0.7020 | **~3.3 s** / query @ top-50 CPU |
| hybrid | Cross-encoder | 0.7016 | **~15.0 s** / query @ top-50 CPU |

Gain framing: hybrid+ColBERT is **+0.035** over hybrid identity, but only **+0.0175** over dense alone. Always show dense in the table.

Latency context: those seconds are **per query**, reranking **50 candidates on CPU**, and previously re-encoded every document on each query. Doc-token precompute (now default in `rerank-ablate`) + smaller pools (try top-20) + `--features cuda` are the intended path to interactive latency. INT8-quantized ColBERT/CE models are a follow-up.

Reproduce ablate (hybrid + dense × identity / MaxSim / CE):

```bash
cargo run -p retrieva-cli --features onnx --release -- bench \
  --suite rerank-ablate --dataset data/beir/scifact --embedder onnx
```

### Load / QPS (ONNX hybrid, Zipf queries, 100 req @ c=4)

| Cache | p50 | hit rate | QPS |
|-------|----:|---------:|----:|
| off | 140 ms | — | **25.7** |
| on | 107 ms | **42%** | **52.3** |

Zipf(s=1.1) over SciFact queries — cache helps when hits exist. Unique round-robin previously showed cache-on *slower* (~0% hits + extra semantic embed); the engine now reuses one query embedding for cache + dense. Hash-path QPS is not resume-ready.

### Other

- Embed parity (batched, multi-length): `cosine_vs_python_min=1.000000` — [`embed_parity_batch.md`](bench/results/embed_parity_batch.md)
- Demo corpus: 8-doc smoke or **3000** SciFact abstracts — [`docs/DEMO_CORPUS.md`](docs/DEMO_CORPUS.md)

Reproduce:

```bash
python scripts/download_bge_small.py
python scripts/download_bge_reranker.py
python scripts/download_colbert.py
python scripts/download_beir.py scifact
bash scripts/run_onnx_remaining.sh   # inside retrieva-onnx:builder
# Optional GPU: cargo run -p retrieva-cli --features "onnx,cuda" -- ...
```

Storage: [docs/STORAGE.md](docs/STORAGE.md). Models: [docs/MODELS.md](docs/MODELS.md).

## Design decisions

| Choice | Decision |
|--------|----------|
| License | Apache-2.0 (patent grant; single LICENSE) |
| Edition / MSRV | Edition 2021, `rust-version = "1.90"` (default builds). `onnx` feature needs rustc ≥ 1.92 (ort) |
| Config | [figment](https://docs.rs/figment) — TOML + `RETRIEVA_*` env |
| Ingest | `.txt` / `.md` / `.markdown` / `.pdf`; word chunks with overlap (`[ingest]`) |
| Sparse index | Tantivy BM25 per collection under `data_dir/indexes/<id>/tantivy` |
| Dense index | usearch (cosine) under `data_dir/indexes/<id>/usearch` + chunk sidecar |
| Doc store | Filesystem JSON under `data_dir/store/` (collections / docs / chunks) |
| WAL | Append-only JSONL + checkpoint under `data_dir/wal/`; replay on open |
| Query cache | Exact (hashed request) + semantic (cosine ≥ threshold); shared query emb with dense; hit-rate stats |
| Fusion | RRF (default) or weighted BM25/dense (`search.fusion`) |
| Embedders | `HashEmbedder` always; `OnnxEmbedder` (`bge-small-en-v1.5`) behind `onnx` |
| ONNX (`ort`) | Exact pin `=2.0.0-rc.15`, shared process-wide Environment; optional `cuda` EP |
| Reranker | ColBERT ONNX MaxSim (doc-token cache) when present; CE ablate on BEIR via `rerank-ablate` |
| REST | axum: `GET /health`, `GET /metrics`, `POST /v1/search`, `POST /v1/ingest` |
| gRPC | tonic `retrieva.v1.Retrieva` on `server.grpc_port` |
| Bench | `--embedder hash\|onnx`; suites matrix / load (Zipf) / `rerank-ablate` |
| Docs | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DESIGN.md](docs/DESIGN.md), [docs/DEMO.md](docs/DEMO.md) |
| CI | `ubuntu-latest`; default features (no onnx) |

## Quick start

```bash
cargo build -p retrieva-cli --release
cargo run -p retrieva-cli -- ingest --collection demo ./demo/docs --embedder hash
cargo run -p retrieva-cli -- query --collection demo "hybrid retrieval" --mode hybrid
cargo run -p retrieva-cli -- serve
make ci
```

**Windows:** Prefer Docker `retrieva-onnx:builder` for `--features onnx` (ort needs glibc ≥ 2.38 / MSVC on native Windows).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
