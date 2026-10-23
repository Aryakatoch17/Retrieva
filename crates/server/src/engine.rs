//! Hybrid retrieval engine: BM25 + dense + fusion (+ optional cache / rerank).

use async_trait::async_trait;
use retrieva_core::traits::{Embedder, Fusion, QueryCache, Reranker, Retriever, SparseIndex};
use retrieva_core::{
    Embedding, LatencyBreakdown, Result, SearchMode, SearchRequest, SearchResponse, Stage,
};
use retrieva_index::{TantivySparseIndex, UsearchVectorIndex};
use std::sync::Arc;
use std::time::Instant;

/// Composes sparse, dense, fusion, cache, and rerank stages.
pub struct HybridEngine<E: Embedder> {
    sparse: Arc<TantivySparseIndex>,
    vectors: Arc<UsearchVectorIndex>,
    embedder: Arc<E>,
    fusion: Arc<dyn Fusion>,
    reranker: Arc<dyn Reranker>,
    cache: Option<Arc<dyn QueryCache>>,
    candidate_pool: usize,
    query_prefix: Option<String>,
}

impl<E: Embedder> HybridEngine<E> {
    /// Build a hybrid engine (no query cache).
    pub fn new(
        sparse: Arc<TantivySparseIndex>,
        vectors: Arc<UsearchVectorIndex>,
        embedder: Arc<E>,
        fusion: Arc<dyn Fusion>,
        reranker: Arc<dyn Reranker>,
        candidate_pool: usize,
    ) -> Self {
        Self {
            sparse,
            vectors,
            embedder,
            fusion,
            reranker,
            cache: None,
            candidate_pool: candidate_pool.max(1),
            query_prefix: None,
        }
    }

    /// Attach a query cache (exact and/or semantic).
    pub fn with_cache(mut self, cache: Arc<dyn QueryCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Optional dense query prefix (BGE instruction).
    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = Some(prefix.into());
        self
    }

    /// Replace the active reranker (e.g. for ablation).
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = reranker;
        self
    }

    /// Shared sparse index (for ingest).
    pub fn sparse(&self) -> Arc<TantivySparseIndex> {
        Arc::clone(&self.sparse)
    }

    /// Shared vector index (for ingest).
    pub fn vectors(&self) -> Arc<UsearchVectorIndex> {
        Arc::clone(&self.vectors)
    }

    /// Shared embedder (for ingest).
    pub fn embedder(&self) -> Arc<E> {
        Arc::clone(&self.embedder)
    }

    /// Shared query cache, if configured.
    pub fn cache(&self) -> Option<Arc<dyn QueryCache>> {
        self.cache.as_ref().map(Arc::clone)
    }

    fn query_text(&self, request: &SearchRequest) -> String {
        match &self.query_prefix {
            Some(prefix) => format!("{prefix}{}", request.query),
            None => request.query.clone(),
        }
    }
}

#[async_trait]
impl<E: Embedder> Retriever for HybridEngine<E> {
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse> {
        request.validate()?;
        let start = Instant::now();
        let mut latency = LatencyBreakdown::default();

        let want_bm25 = matches!(
            request.mode,
            SearchMode::Bm25 | SearchMode::Hybrid | SearchMode::HybridRerank
        );
        let want_dense = matches!(
            request.mode,
            SearchMode::Dense | SearchMode::Hybrid | SearchMode::HybridRerank
        );
        let want_rerank = matches!(request.mode, SearchMode::HybridRerank);
        let semantic = self
            .cache
            .as_ref()
            .is_some_and(|c| c.uses_query_embedding());

        // Exact-only probe first: Zipf / repeated queries must not pay an embed on hit.
        if request.use_cache {
            if let Some(cache) = &self.cache {
                let t0 = Instant::now();
                if let Some(mut hit) = cache.get_with_query_embedding(&request, None).await? {
                    latency.record(Stage::Cache, t0.elapsed());
                    latency.record(Stage::Total, start.elapsed());
                    hit.cache_hit = true;
                    hit.latency = latency;
                    return Ok(hit);
                }
                latency.record(Stage::Cache, t0.elapsed());
            }
        }

        let mut query_emb: Option<Embedding> = None;
        if want_dense || (request.use_cache && semantic) {
            let t0 = Instant::now();
            let query_text = self.query_text(&request);
            query_emb = Some(self.embedder.embed(&query_text).await?);
            latency.record(Stage::Embed, t0.elapsed());
        }
        let emb_slice = query_emb.as_ref().map(|e| e.values.as_slice());

        // Semantic lookup reuses the dense query embedding (no second encode).
        if request.use_cache && semantic {
            if let Some(cache) = &self.cache {
                let t0 = Instant::now();
                if let Some(mut hit) = cache.get_with_query_embedding(&request, emb_slice).await? {
                    latency.record(Stage::Cache, t0.elapsed());
                    latency.record(Stage::Total, start.elapsed());
                    hit.cache_hit = true;
                    hit.latency = latency;
                    return Ok(hit);
                }
                latency.record(Stage::Cache, t0.elapsed());
            }
        }

        let pool = self.candidate_pool.max(request.top_k);

        let mut bm25_hits = Vec::new();
        if want_bm25 {
            let t0 = Instant::now();
            bm25_hits = self
                .sparse
                .search(&request.collection_id, &request.query, pool)
                .await?;
            latency.record(Stage::Bm25, t0.elapsed());
        }

        let mut dense_hits = Vec::new();
        if want_dense {
            let embedding = query_emb
                .as_ref()
                .expect("query embedding prepared for dense search");
            let t1 = Instant::now();
            dense_hits = self
                .vectors
                .search_chunks(&request.collection_id, &embedding.values, pool)
                .await?;
            latency.record(Stage::Dense, t1.elapsed());
        }

        let t_fuse = Instant::now();
        let mut results = match request.mode {
            SearchMode::Bm25 => bm25_hits,
            SearchMode::Dense => dense_hits,
            SearchMode::Hybrid | SearchMode::HybridRerank => {
                let fuse_k = if want_rerank {
                    request.rerank_top_n.max(request.top_k)
                } else {
                    request.top_k
                };
                self.fusion.fuse(&[bm25_hits, dense_hits], fuse_k)?
            }
        };
        if matches!(request.mode, SearchMode::Hybrid | SearchMode::HybridRerank) {
            latency.record(Stage::Fusion, t_fuse.elapsed());
        }

        if want_rerank {
            let t0 = Instant::now();
            results = self
                .reranker
                .rerank(&request.query, &results, request.top_k)
                .await?;
            latency.record(Stage::Rerank, t0.elapsed());
        } else {
            results.truncate(request.top_k);
        }

        latency.record(Stage::Total, start.elapsed());
        let response = SearchResponse {
            results,
            latency,
            cache_hit: false,
        };

        if request.use_cache {
            if let Some(cache) = &self.cache {
                cache
                    .put_with_query_embedding(&request, &response, emb_slice)
                    .await?;
            }
        }

        Ok(response)
    }
}
