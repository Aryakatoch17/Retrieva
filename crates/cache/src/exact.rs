//! In-memory exact-match query cache (LRU + optional TTL).

use crate::key::exact_key;
use async_trait::async_trait;
use parking_lot::Mutex;
use retrieva_core::traits::QueryCache;
use retrieva_core::{CacheStats, CollectionId, Result, SearchRequest, SearchResponse};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

struct Entry {
    collection_id: String,
    response: SearchResponse,
    expires_at: Option<Instant>,
}

/// Exact-match cache keyed by hashed request identity.
pub struct ExactQueryCache {
    capacity: usize,
    ttl: Option<Duration>,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
}

struct Inner {
    map: HashMap<String, Entry>,
    order: VecDeque<String>,
}

impl ExactQueryCache {
    /// Create a cache with `capacity` entries and optional TTL (`None` / zero = no expiry).
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            ttl: if ttl_secs == 0 {
                None
            } else {
                Some(Duration::from_secs(ttl_secs))
            },
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn expired(entry: &Entry, now: Instant) -> bool {
        entry.expires_at.is_some_and(|t| now >= t)
    }
}

#[async_trait]
impl QueryCache for ExactQueryCache {
    async fn get_with_query_embedding(
        &self,
        request: &SearchRequest,
        _query_embedding: Option<&[f32]>,
    ) -> Result<Option<SearchResponse>> {
        let key = exact_key(request);
        let mut guard = self.inner.lock();
        let now = Instant::now();
        let hit = match guard.map.get(&key) {
            Some(entry) if Self::expired(entry, now) => {
                guard.map.remove(&key);
                guard.order.retain(|k| k != &key);
                None
            }
            Some(entry) => {
                let mut response = entry.response.clone();
                response.cache_hit = true;
                Some(response)
            }
            None => None,
        };
        if hit.is_some() {
            if let Some(pos) = guard.order.iter().position(|k| k == &key) {
                guard.order.remove(pos);
            }
            guard.order.push_back(key.clone());
            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!(%key, "exact cache hit");
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hit)
    }

    async fn put_with_query_embedding(
        &self,
        request: &SearchRequest,
        response: &SearchResponse,
        _query_embedding: Option<&[f32]>,
    ) -> Result<()> {
        let key = exact_key(request);
        let now = Instant::now();
        let entry = Entry {
            collection_id: request.collection_id.as_str().to_string(),
            response: {
                let mut r = response.clone();
                r.cache_hit = false;
                r
            },
            expires_at: self.ttl.map(|d| now + d),
        };
        let mut guard = self.inner.lock();
        if guard.map.contains_key(&key) {
            guard.order.retain(|k| k != &key);
        }
        guard.map.insert(key.clone(), entry);
        guard.order.push_back(key);
        while guard.map.len() > self.capacity {
            if let Some(old) = guard.order.pop_front() {
                guard.map.remove(&old);
            } else {
                break;
            }
        }
        Ok(())
    }

    async fn invalidate_collection(&self, collection_id: &CollectionId) -> Result<()> {
        let id = collection_id.as_str();
        let mut guard = self.inner.lock();
        let doomed: Vec<String> = guard
            .map
            .iter()
            .filter(|(_, e)| e.collection_id == id)
            .map(|(k, _)| k.clone())
            .collect();
        for key in doomed {
            guard.map.remove(&key);
            guard.order.retain(|k| k != &key);
        }
        Ok(())
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

    fn req(q: &str) -> SearchRequest {
        SearchRequest {
            collection_id: CollectionId::new("c"),
            query: q.into(),
            top_k: 5,
            mode: SearchMode::Bm25,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: true,
        }
    }

    fn empty_response() -> SearchResponse {
        SearchResponse {
            results: vec![],
            latency: LatencyBreakdown::default(),
            cache_hit: false,
        }
    }

    #[tokio::test]
    async fn hit_miss_invalidate() {
        let cache = ExactQueryCache::new(10, 0);
        let r = req("hello");
        assert!(cache.get(&r).await.unwrap().is_none());
        cache.put(&r, &empty_response()).await.unwrap();
        assert!(cache.get(&r).await.unwrap().unwrap().cache_hit);
        cache
            .invalidate_collection(&CollectionId::new("c"))
            .await
            .unwrap();
        assert!(cache.get(&r).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn lru_evicts() {
        let cache = ExactQueryCache::new(2, 0);
        cache.put(&req("a"), &empty_response()).await.unwrap();
        cache.put(&req("b"), &empty_response()).await.unwrap();
        // Touch a so b is older... actually order: a, b. Put c evicts a.
        cache.put(&req("c"), &empty_response()).await.unwrap();
        assert!(cache.get(&req("a")).await.unwrap().is_none());
        assert!(cache.get(&req("b")).await.unwrap().is_some());
        assert!(cache.get(&req("c")).await.unwrap().is_some());
    }
}
