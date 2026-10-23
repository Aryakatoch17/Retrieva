//! Stable cache keys derived from [`SearchRequest`].

use retrieva_core::SearchRequest;
use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Hex SHA-256 of the request fields that affect result identity.
pub fn exact_key(request: &SearchRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.collection_id.as_str().as_bytes());
    hasher.update(b"\0");
    hasher.update(request.query.trim().as_bytes());
    hasher.update(b"\0");
    hasher.update(request.top_k.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(format!("{:?}", request.mode).as_bytes());
    hasher.update(b"\0");
    hasher.update(request.rerank_top_n.to_le_bytes());
    hasher.update(b"\0");
    let mut keys: Vec<_> = request.filters.metadata_eq.iter().collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in keys {
        hasher.update(k.as_bytes());
        hasher.update(b"=");
        hasher.update(v.as_bytes());
        hasher.update(b"\0");
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Fingerprint of non-query fields (collection, mode, top_k, filters) for semantic lookup.
pub fn semantic_fingerprint(request: &SearchRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.collection_id.as_str().as_bytes());
    hasher.update(b"\0");
    hasher.update(request.top_k.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(format!("{:?}", request.mode).as_bytes());
    hasher.update(b"\0");
    hasher.update(request.rerank_top_n.to_le_bytes());
    hasher.update(b"\0");
    let mut keys: Vec<_> = request.filters.metadata_eq.iter().collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in keys {
        hasher.update(k.as_bytes());
        hasher.update(b"=");
        hasher.update(v.as_bytes());
        hasher.update(b"\0");
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{CollectionId, SearchFilters, SearchMode};

    fn req(q: &str) -> SearchRequest {
        SearchRequest {
            collection_id: CollectionId::new("c"),
            query: q.into(),
            top_k: 10,
            mode: SearchMode::Hybrid,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: true,
        }
    }

    #[test]
    fn exact_key_stable_and_sensitive() {
        assert_eq!(exact_key(&req("hello")), exact_key(&req("hello")));
        assert_ne!(exact_key(&req("hello")), exact_key(&req("Hello")));
        let mut a = req("q");
        let mut b = req("q");
        a.top_k = 5;
        b.top_k = 10;
        assert_ne!(exact_key(&a), exact_key(&b));
    }

    #[test]
    fn semantic_fingerprint_ignores_query_text() {
        assert_eq!(
            semantic_fingerprint(&req("alpha")),
            semantic_fingerprint(&req("beta"))
        );
    }
}
