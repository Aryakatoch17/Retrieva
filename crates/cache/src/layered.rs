//! Exact-then-semantic layered [`QueryCache`].

use crate::exact::ExactQueryCache;
use crate::semantic::SemanticQueryCache;
use async_trait::async_trait;
use retrieva_core::traits::{Embedder, QueryCache};
use retrieva_core::{CacheConfig, CacheStats, CollectionId, Result, SearchRequest, SearchResponse};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::debug;

/// Combines exact and optional semantic caches per [`CacheConfig`].
pub struct LayeredQueryCache<E: Embedder> {
    exact: Option<ExactQueryCache>,
    semantic: Option<SemanticQueryCache<E>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<E: Embedder + 'static> LayeredQueryCache<E> {
    /// Build from config + embedder (embedder used only when semantic is enabled).
    pub fn from_config(config: &CacheConfig, embedder: Arc<E>) -> Self {
        let capacity = config.exact_capacity.max(1) as usize;
        let exact = config
            .exact_enabled
            .then(|| ExactQueryCache::new(capacity, config.ttl_secs));
        let semantic = config.semantic_enabled.then(|| {
            SemanticQueryCache::new(
                embedder,
                config.semantic_threshold,
                capacity,
                config.ttl_secs,
            )
        });
        Self {
            exact,
            semantic,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl<E: Embedder + 'static> QueryCache for LayeredQueryCache<E> {
    async fn get_with_query_embedding(
        &self,
        request: &SearchRequest,
        query_embedding: Option<&[f32]>,
    ) -> Result<Option<SearchResponse>> {
        if !request.use_cache {
            return Ok(None);
        }
        // `None` → exact only (encode-free). `Some(emb)` → semantic only, reusing that emb.
        // The engine probes exact first, then encodes once, then probes semantic.
        match query_embedding {
            None => {
                if let Some(exact) = &self.exact {
                    if let Some(hit) = exact
                        .get_with_query_embedding(request, None)
                        .await?
                    {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        debug!("layered cache: exact hit");
                        return Ok(Some(hit));
                    }
                }
                // Not a layered miss yet if semantic may still be consulted with an emb.
                if self.semantic.is_none() {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                }
                Ok(None)
            }
            Some(_) => {
                if let Some(semantic) = &self.semantic {
                    if let Some(hit) = semantic
                        .get_with_query_embedding(request, query_embedding)
                        .await?
                    {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        debug!("layered cache: semantic hit");
                        return Ok(Some(hit));
                    }
                }
                self.misses.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    async fn put_with_query_embedding(
        &self,
        request: &SearchRequest,
        response: &SearchResponse,
        query_embedding: Option<&[f32]>,
    ) -> Result<()> {
        if !request.use_cache {
            return Ok(());
        }
        if let Some(exact) = &self.exact {
            exact
                .put_with_query_embedding(request, response, query_embedding)
                .await?;
        }
        if let Some(semantic) = &self.semantic {
            semantic
                .put_with_query_embedding(request, response, query_embedding)
                .await?;
        }
        Ok(())
    }

    async fn invalidate_collection(&self, collection_id: &CollectionId) -> Result<()> {
        if let Some(exact) = &self.exact {
            exact.invalidate_collection(collection_id).await?;
        }
        if let Some(semantic) = &self.semantic {
            semantic.invalidate_collection(collection_id).await?;
        }
        Ok(())
    }

    fn uses_query_embedding(&self) -> bool {
        self.semantic.is_some()
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        if let Some(exact) = &self.exact {
            exact.reset_stats();
        }
        if let Some(semantic) = &self.semantic {
            semantic.reset_stats();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{LatencyBreakdown, SearchFilters, SearchMode};
    use retrieva_ingest::HashEmbedder;

    #[tokio::test]
    async fn exact_preferred_over_semantic() {
        let emb = Arc::new(HashEmbedder::new(16));
        let cfg = CacheConfig {
            exact_enabled: true,
            semantic_enabled: true,
            exact_capacity: 100,
            semantic_threshold: 0.5,
            ttl_secs: 0,
        };
        let cache = LayeredQueryCache::from_config(&cfg, emb);
        let req = SearchRequest {
            collection_id: CollectionId::new("c"),
            query: "layered".into(),
            top_k: 3,
            mode: SearchMode::Bm25,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: true,
        };
        let resp = SearchResponse {
            results: vec![],
            latency: LatencyBreakdown::default(),
            cache_hit: false,
        };
        cache.put(&req, &resp).await.unwrap();
        assert!(cache.get(&req).await.unwrap().unwrap().cache_hit);
        cache
            .invalidate_collection(&CollectionId::new("c"))
            .await
            .unwrap();
        assert!(cache.get(&req).await.unwrap().is_none());
    }
}
