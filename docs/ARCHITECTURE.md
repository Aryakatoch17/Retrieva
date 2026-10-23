# Retrieva architecture

Retrieva is a hybrid RAG retrieval engine: one process owns ingest, indexes, cache, and query APIs.

## Crate map

```
retrieva-cli ──► retrieva-server ──► HybridEngine
                     │                    │
                     │                    ├── TantivySparseIndex (BM25)
                     │                    ├── UsearchVectorIndex (ANN)
                     │                    ├── Embedder (hash | onnx)
                     │                    ├── Fusion (RRF | weighted)
                     │                    ├── Reranker (MaxSim | CE | identity)
                     │                    └── QueryCache (exact + semantic)
                     │
                     └── IncrementalIndex ──► WAL + FsDocStore + sparse + dense
```

| Crate | Responsibility |
|-------|----------------|
| `retrieva-core` | Types, traits, config, errors, logging |
| `retrieva-ingest` | Parse → chunk → embed |
| `retrieva-index` | Tantivy, usearch, FsDocStore, WAL, IncrementalIndex |
| `retrieva-fusion` | RRF / weighted fusion |
| `retrieva-rerank` | MaxSim, cross-encoder, ablation |
| `retrieva-cache` | Exact + semantic query cache |
| `retrieva-server` | axum REST, tonic gRPC, auth, rate limit, metrics |
| `retrieva-bench` | BEIR loaders, nDCG/Recall/MRR, load test |
| `retrieva-cli` | `retrieva` binary |

## Data layout (`data_dir`)

```
data_dir/
  wal/                 log.jsonl + checkpoint.json
  store/               collections / documents / chunks (JSON)
  indexes/<col>/
    tantivy/           BM25 index
    usearch/           ANN index + chunks sidecar
```

## Query path

1. Optional exact cache lookup (hashed request key).
2. Optional semantic cache (cosine ≥ threshold on query embedding).
3. BM25 and/or dense retrieval up to `candidate_pool`.
4. Fusion when mode is hybrid / hybrid+rerank.
5. Optional MaxSim (or cross-encoder) rerank; truncate to `top_k`.
6. Store response in cache when `use_cache` is true.

## Write path

1. Append op to WAL (fsync).
2. Apply to FsDocStore + Tantivy + usearch.
3. Checkpoint WAL sequence.
4. Invalidate query cache for the collection.

On open, `IncrementalIndex` replays any entries after the last checkpoint.

## APIs

| Surface | Endpoints / RPCs |
|---------|------------------|
| REST | `GET /health`, `GET /metrics`, `POST /v1/search`, `POST /v1/ingest` |
| gRPC | `retrieva.v1.Retrieva` — Health, Search, Ingest (`proto/retrieva/v1/retrieva.proto`) |

Auth (optional API key) and rate limiting wrap both surfaces. Prometheus text is exposed at `/metrics`.

## Deployment sketch

```
Client ──HTTP/gRPC──► Retrieva (:8080 / :50051)
                         │
                         ├── volume: data_dir
                         └── scrape: Prometheus (:9090) [compose profile metrics]
```

See [DESIGN.md](DESIGN.md) for trade-offs and [../docker-compose.yml](../docker-compose.yml) for the demo stack.
