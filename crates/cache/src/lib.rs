//! Exact-match and semantic query caches (Phase 7).
//!
//! [`LayeredQueryCache`] tries exact key lookup first, then embedding cosine similarity.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod exact;
mod key;
mod layered;
mod semantic;

pub use exact::ExactQueryCache;
pub use key::{exact_key, semantic_fingerprint};
pub use layered::LayeredQueryCache;
pub use semantic::SemanticQueryCache;

use retrieva_core::traits::{Embedder, QueryCache};
use retrieva_core::CacheConfig;
use std::sync::Arc;

/// Build a dyn [`QueryCache`] from config (exact and/or semantic).
pub fn cache_from_config<E: Embedder + 'static>(
    config: &CacheConfig,
    embedder: Arc<E>,
) -> Arc<dyn QueryCache> {
    Arc::new(LayeredQueryCache::from_config(config, embedder))
}
