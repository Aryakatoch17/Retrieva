//! Core extension traits. Concrete backends live in sibling crates.
//!
//! Everything is behind traits so tests can use [`crate::mocks`] without
//! pulling in RocksDB, Tantivy, usearch, or ONNX Runtime.

use crate::types::{
    CacheStats, Chunk, ChunkId, CollectionId, CollectionInfo, Document, DocumentId, Embedding,
    Neighbor, ScoredChunk, SearchRequest, SearchResponse,
};
use crate::Result;
use async_trait::async_trait;

/// Produces dense embeddings for queries and documents.
///
/// Production implementations are gated behind the `onnx` feature in
/// `retrieva-ingest` / `retrieva-rerank`. Core and BM25 paths never depend on it.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embedding dimensionality.
    fn dimensions(&self) -> usize;

    /// Embed a single text.
    async fn embed(&self, text: &str) -> Result<Embedding>;

    /// Embed a batch of texts (preferred path for ingest throughput).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
}

/// Approximate nearest-neighbor vector index.
///
/// Default production backend: usearch. Alternative backends (e.g. hnsw_rs)
/// can be added later without changing call sites.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Insert or replace a vector for `key`.
    async fn upsert(&self, key: u64, vector: &[f32]) -> Result<()>;

    /// Remove a vector by key.
    async fn remove(&self, key: u64) -> Result<()>;

    /// Search for the `top_k` nearest neighbors of `query`.
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Neighbor>>;

    /// Number of indexed vectors.
    async fn len(&self) -> Result<usize>;

    /// Whether the index is empty.
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }
}

/// Sparse / lexical index (BM25 via Tantivy in production).
#[async_trait]
pub trait SparseIndex: Send + Sync {
    /// Index or replace a chunk.
    async fn upsert(&self, chunk: &Chunk) -> Result<()>;

    /// Remove a chunk by id.
    async fn remove(&self, chunk_id: &ChunkId) -> Result<()>;

    /// BM25 search.
    async fn search(
        &self,
        collection_id: &CollectionId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>>;
}

/// Document / chunk metadata store.
#[async_trait]
pub trait DocStore: Send + Sync {
    /// Create a collection.
    async fn create_collection(&self, info: &CollectionInfo) -> Result<()>;

    /// Delete a collection and all of its documents.
    async fn delete_collection(&self, id: &CollectionId) -> Result<()>;

    /// Fetch collection metadata.
    async fn get_collection(&self, id: &CollectionId) -> Result<Option<CollectionInfo>>;

    /// Upsert documents (chunking happens in ingest).
    async fn upsert_documents(&self, docs: &[Document]) -> Result<()>;

    /// Delete a document and its chunks.
    async fn delete_document(&self, id: &DocumentId) -> Result<()>;

    /// Fetch a chunk by id.
    async fn get_chunk(&self, id: &ChunkId) -> Result<Option<Chunk>>;

    /// Persist chunks produced by ingest.
    async fn put_chunks(&self, chunks: &[Chunk]) -> Result<()>;

    /// List chunk ids belonging to a document (for incremental deletes).
    async fn list_chunks_for_document(&self, id: &DocumentId) -> Result<Vec<ChunkId>>;

    /// Delete a single chunk and update the document's chunk index.
    async fn delete_chunk(&self, id: &ChunkId) -> Result<()>;
}

/// Reranks a candidate list given the original query.
///
/// Phase 4 ships a cross-encoder; Phase 5 adds MaxSim and ablates the two.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `candidates` for `query`, returning at most `top_k` results.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[ScoredChunk],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>>;
}

/// End-to-end retriever composing sparse, dense, fusion, cache, and rerank.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Execute a search request.
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse>;
}

/// Fusion strategy over ranked lists from multiple retrievers.
pub trait Fusion: Send + Sync {
    /// Fuse one or more ranked lists into a single list of length `top_k`.
    fn fuse(&self, lists: &[Vec<ScoredChunk>], top_k: usize) -> Result<Vec<ScoredChunk>>;
}

/// Exact-match and/or semantic query cache.
#[async_trait]
pub trait QueryCache: Send + Sync {
    /// Look up a cached response.
    async fn get(&self, request: &SearchRequest) -> Result<Option<SearchResponse>> {
        self.get_with_query_embedding(request, None).await
    }

    /// Lookup using a precomputed query embedding (semantic caches skip a second encode).
    async fn get_with_query_embedding(
        &self,
        request: &SearchRequest,
        query_embedding: Option<&[f32]>,
    ) -> Result<Option<SearchResponse>>;

    /// Store a response.
    async fn put(&self, request: &SearchRequest, response: &SearchResponse) -> Result<()> {
        self.put_with_query_embedding(request, response, None).await
    }

    /// Store using a precomputed query embedding (semantic caches skip a second encode).
    async fn put_with_query_embedding(
        &self,
        request: &SearchRequest,
        response: &SearchResponse,
        query_embedding: Option<&[f32]>,
    ) -> Result<()>;

    /// Invalidate entries affected by writes to `collection_id`.
    async fn invalidate_collection(&self, collection_id: &CollectionId) -> Result<()>;

    /// Whether lookups may encode the query (semantic layer).
    fn uses_query_embedding(&self) -> bool {
        false
    }

    /// Cumulative hit/miss counters since last reset.
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }

    /// Zero hit/miss counters.
    fn reset_stats(&self) {}
}
