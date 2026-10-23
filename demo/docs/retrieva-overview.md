# Retrieva overview

Retrieva is a self-hostable hybrid RAG retrieval engine written in Rust.
It combines BM25 sparse search (Tantivy) with dense approximate nearest-neighbor
search (usearch), optional MaxSim late-interaction reranking, and a write-ahead
log for incremental indexing.

## Hybrid retrieval

Sparse BM25 catches exact keyword matches. Dense embeddings catch paraphrases.
Reciprocal Rank Fusion (RRF) merges the two ranked lists before an optional
reranker. Exact and semantic query caches reduce repeated-query latency.

## When to use it

Use Retrieva when you want a single binary (or container) that exposes REST and
gRPC search/ingest APIs, Prometheus metrics, and reproducible BEIR evaluation
without depending on a managed vector database.
