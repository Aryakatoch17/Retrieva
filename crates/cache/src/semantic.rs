//! Embedding-similarity semantic query cache.

use crate::key::semantic_fingerprint;
use async_trait::async_trait;
use parking_lot::Mutex;
use retrieva_core::traits::{Embedder, QueryCache};
use retrieva_core::{CacheStats, CollectionId, Result, SearchRequest, SearchResponse};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;

struct Entry {
    collection_id: String,
    fingerprint: String,
    embedding: Vec<f32>,
    response: SearchResponse,
    expires_at: Option<Instant>,
    inserted_at: Instant,
}

/// Semantic cache: cosine similarity over query embeddings within a collection fingerprint.
pub struct SemanticQueryCache<E: Embedder> {
    embedder: Arc<E>,
    threshold: f32,
    capacity: usize,
    ttl: Option<Duration>,
    inner: Mutex<Vec<Entry>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<E: Embedder> SemanticQueryCache<E> {
    /// Build a semantic cache.
    pub fn new(embedder: Arc<E>, threshold: f32, capacity: usize, ttl_secs: u64) -> Self {
        Self {
            embedder,
            threshold: threshold.clamp(0.0, 1.0),
            capacity: capacity.max(1),
            ttl: if ttl_secs == 0 {
                None
            } else {
                Some(Duration::from_secs(ttl_secs))
            },
            inner: Mutex::new(Vec::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    async fn resolve_embedding(
        &self,
        request: &SearchRequest,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if let Some(v) = query_embedding {
            return Ok(v.to_vec());
        }
        Ok(self.embedder.embed(request.query.trim()).await?.values)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[async_trait]
impl<E: Embedder + 'static> QueryCache for SemanticQueryCache<E> {
    async fn get_with_query_embedding(
        &self,
        request: &SearchRequest,
        query_embedding: Option<&[f32]>,
    ) -> Result<Option<SearchResponse>> {
        let emb = self.resolve_embedding(request, query_embedding).await?;
        let fp = semantic_fingerprint(request);
        let col = request.collection_id.as_str();
        let now = Instant::now();

        let mut guard = self.inner.lock();
        guard.retain(|e| e.expires_at.map(|t| now < t).unwrap_or(true));

        let mut best: Option<(usize, f32)> = None;
        for (i, entry) in guard.iter().enumerate() {
            if entry.collection_id != col || entry.fingerprint != fp {
                continue;
            }
            let sim = cosine(&emb, &entry.embedding);
            if sim >= self.threshold {
                match best {
                    Some((_, s)) if s >= sim => {}
                    _ => best = Some((i, sim)),
                }
            }
        }
        if let Some((idx, sim)) = best {
            let mut response = guard[idx].response.clone();
            response.cache_hit = true;
            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!(sim, threshold = self.threshold, "semantic cache hit");
            return Ok(Some(response));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    async fn put_with_query_embedding(
        &self,
        request: &SearchRequest,
        response: &SearchResponse,
        query_embedding: Option<&[f32]>,
    ) -> Result<()> {
        let emb = self.resolve_embedding(request, query_embedding).await?;
        let now = Instant::now();
        let entry = Entry {
            collection_id: request.collection_id.as_str().to_string(),
            fingerprint: semantic_fingerprint(request),
            embedding: emb,
            response: {
                let mut r = response.clone();
                r.cache_hit = false;
                r
            },
            expires_at: self.ttl.map(|d| now + d),
            inserted_at: now,
        };
        let mut guard = self.inner.lock();
        guard.push(entry);
        while guard.len() > self.capacity {
            // Evict oldest.
            if let Some(min_i) = guard
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(i, _)| i)
            {
                guard.remove(min_i);
            } else {
                break;
            }
        }
        Ok(())
    }

    async fn invalidate_collection(&self, collection_id: &CollectionId) -> Result<()> {
        let id = collection_id.as_str();
        self.inner.lock().retain(|e| e.collection_id != id);
        Ok(())
    }

    fn uses_query_embedding(&self) -> bool {
        true
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{LatencyBreakdown, SearchFilters, SearchMode};
    use retrieva_ingest::HashEmbedder;

    fn req(q: &str) -> SearchRequest {
        SearchRequest {
            collection_id: CollectionId::new("c"),
            query: q.into(),
            top_k: 5,
            mode: SearchMode::Dense,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: true,
        }
    }

    #[tokio::test]
    async fn similar_query_hits() {
        let emb = Arc::new(HashEmbedder::new(32));
        let cache = SemanticQueryCache::new(emb, 0.99, 16, 0);
        let r = req("semantic cache probe alpha");
        let resp = SearchResponse {
            results: vec![],
            latency: LatencyBreakdown::default(),
            cache_hit: false,
        };
        cache.put(&r, &resp).await.unwrap();
        // Identical query must hit.
        assert!(cache.get(&r).await.unwrap().is_some());
        // Unrelated query should miss at high threshold.
        assert!(cache
            .get(&req("zzzz totally different vocabulary xyzzy"))
            .await
            .unwrap()
            .is_none());
        let s = cache.stats();
        assert!(s.hits >= 1);
        assert!(s.misses >= 1);
    }

    #[tokio::test]
    async fn reused_embedding_skips_reencode() {
        let emb = Arc::new(HashEmbedder::new(32));
        let cache = SemanticQueryCache::new(Arc::clone(&emb), 0.99, 16, 0);
        let r = req("reuse emb");
        let vector = emb.embed(r.query.trim()).await.unwrap().values;
        let resp = SearchResponse {
            results: vec![],
            latency: LatencyBreakdown::default(),
            cache_hit: false,
        };
        cache
            .put_with_query_embedding(&r, &resp, Some(&vector))
            .await
            .unwrap();
        assert!(cache
            .get_with_query_embedding(&r, Some(&vector))
            .await
            .unwrap()
            .is_some());
    }
}
