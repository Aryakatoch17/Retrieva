//! BM25 and dense retrievers for Phase 2 / 3 CLI queries.

use crate::sparse::TantivySparseIndex;
use crate::vector::UsearchVectorIndex;
use async_trait::async_trait;
use retrieva_core::traits::{Embedder, Retriever, SparseIndex};
use retrieva_core::{
    LatencyBreakdown, Result, RetrievaError, SearchMode, SearchRequest, SearchResponse, Stage,
};
use std::sync::Arc;
use std::time::Instant;

/// Sparse-only retriever wrapping [`TantivySparseIndex`].
pub struct Bm25Retriever {
    sparse: Arc<TantivySparseIndex>,
}

impl Bm25Retriever {
    /// Create a retriever over an existing sparse index.
    pub fn new(sparse: Arc<TantivySparseIndex>) -> Self {
        Self { sparse }
    }

    /// Open a retriever rooted at `data_dir`.
    pub fn open(data_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        Ok(Self::new(Arc::new(TantivySparseIndex::open(data_dir)?)))
    }

    /// Borrow the underlying sparse index (for ingest).
    pub fn sparse(&self) -> &TantivySparseIndex {
        &self.sparse
    }

    /// Shared ownership of the sparse index.
    pub fn sparse_arc(&self) -> Arc<TantivySparseIndex> {
        Arc::clone(&self.sparse)
    }
}

#[async_trait]
impl Retriever for Bm25Retriever {
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse> {
        request.validate()?;
        if !matches!(request.mode, SearchMode::Bm25) {
            return Err(RetrievaError::InvalidArgument(format!(
                "Bm25Retriever supports SearchMode::Bm25 only, got {:?}",
                request.mode
            )));
        }

        let start = Instant::now();
        let mut latency = LatencyBreakdown::default();

        let bm25_start = Instant::now();
        let results = self
            .sparse
            .search(&request.collection_id, &request.query, request.top_k)
            .await?;
        latency.record(Stage::Bm25, bm25_start.elapsed());
        latency.record(Stage::Total, start.elapsed());

        Ok(SearchResponse {
            results,
            latency,
            cache_hit: false,
        })
    }
}

/// Dense ANN retriever: embed query → usearch → scored chunks.
pub struct DenseRetriever<E: Embedder> {
    embedder: Arc<E>,
    vectors: Arc<UsearchVectorIndex>,
    /// Optional BGE-style query prefix applied before embedding.
    query_prefix: Option<String>,
}

impl<E: Embedder> DenseRetriever<E> {
    /// Create a dense retriever.
    pub fn new(embedder: Arc<E>, vectors: Arc<UsearchVectorIndex>) -> Self {
        Self {
            embedder,
            vectors,
            query_prefix: None,
        }
    }

    /// Set a query instruction prefix (e.g. BGE retrieval prompt).
    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = Some(prefix.into());
        self
    }
}

#[async_trait]
impl<E: Embedder> Retriever for DenseRetriever<E> {
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse> {
        request.validate()?;
        if !matches!(request.mode, SearchMode::Dense) {
            return Err(RetrievaError::InvalidArgument(format!(
                "DenseRetriever supports SearchMode::Dense only, got {:?}",
                request.mode
            )));
        }

        let start = Instant::now();
        let mut latency = LatencyBreakdown::default();

        let embed_start = Instant::now();
        let query_text = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{}", request.query),
            None => request.query.clone(),
        };
        let embedding = self.embedder.embed(&query_text).await?;
        latency.record(Stage::Embed, embed_start.elapsed());

        let dense_start = Instant::now();
        let results = self
            .vectors
            .search_chunks(&request.collection_id, &embedding.values, request.top_k)
            .await?;
        latency.record(Stage::Dense, dense_start.elapsed());
        latency.record(Stage::Total, start.elapsed());

        Ok(SearchResponse {
            results,
            latency,
            cache_hit: false,
        })
    }
}
