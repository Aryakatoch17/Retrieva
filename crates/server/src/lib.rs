//! HTTP and gRPC servers for Retrieva.
//!
//! Phase 4: axum REST (`/health`, `/v1/search`, `/v1/ingest`).
//! Phase 5: MaxSim-primary rerank via config.
//! Phase 6: WAL-backed incremental ingest via [`IncrementalIndex`].
//! Phase 7: exact + semantic query cache with invalidation.
//! Phase 8: gRPC, API-key auth, rate limits, Prometheus `/metrics`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod auth;
mod engine;
mod grpc;
mod metrics;
mod ratelimit;
mod rest;

pub use engine::HybridEngine;
pub use rest::router;

use crate::metrics::Metrics;
use crate::ratelimit::{build_limiter, SharedLimiter};
use retrieva_cache::cache_from_config;
use retrieva_core::traits::{Fusion, QueryCache};
use retrieva_core::{Config, Result, RetrievaError};
use retrieva_fusion::fusion_from_config;
use retrieva_index::IncrementalIndex;
use retrieva_ingest::HashEmbedder;
use retrieva_rerank::reranker_from_config;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    /// Resolved configuration.
    pub config: Arc<Config>,
    /// Hybrid retrieval engine.
    pub engine: Arc<HybridEngine<HashEmbedder>>,
    /// WAL-backed incremental index (ingest / delete / recover).
    pub index: Arc<IncrementalIndex>,
    /// Shared query cache (invalidated on ingest / delete).
    pub cache: Arc<dyn QueryCache>,
    /// Prometheus metrics.
    pub metrics: Arc<Metrics>,
    /// Optional shared rate limiter.
    pub limiter: SharedLimiter,
}

impl AppState {
    /// Open indexes under `config.data_dir`, replay WAL, and build engine from config.
    pub async fn open_default(config: Config) -> Result<Self> {
        let config = Arc::new(config);
        std::fs::create_dir_all(&config.data_dir)?;

        let index =
            Arc::new(IncrementalIndex::open(&config.data_dir, config.embedding.dimensions).await?);
        let embedder = Arc::new(HashEmbedder::new(config.embedding.dimensions));
        let cache = cache_from_config(&config.cache, Arc::clone(&embedder));
        let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&config.search)?);
        let reranker = reranker_from_config(&config.rerank)?;
        let engine = Arc::new(
            HybridEngine::new(
                index.sparse(),
                index.vectors(),
                embedder,
                fusion,
                reranker,
                config.search.candidate_pool,
            )
            .with_cache(Arc::clone(&cache)),
        );
        let metrics =
            Arc::new(Metrics::new().map_err(|e| RetrievaError::Internal(format!("metrics: {e}")))?);
        let limiter = build_limiter(config.server.rate_limit_rps);

        Ok(Self {
            config,
            engine,
            index,
            cache,
            metrics,
            limiter,
        })
    }
}

/// Serve REST + gRPC until either side exits.
pub async fn serve(config: Config) -> Result<()> {
    let rest_addr = SocketAddr::from((parse_host(&config.server.host)?, config.server.port));
    let grpc_addr = SocketAddr::from((parse_host(&config.server.host)?, config.server.grpc_port));
    let api_key = config.server.api_key.clone();

    let state = AppState::open_default(config).await?;
    let app = router(state.clone());

    info!(%rest_addr, "retrieva REST listening");
    info!(%grpc_addr, metrics = "/metrics", "retrieva gRPC + Prometheus");

    let listener = tokio::net::TcpListener::bind(rest_addr)
        .await
        .map_err(RetrievaError::Io)?;

    let rest = async {
        axum::serve(listener, app)
            .await
            .map_err(|e| RetrievaError::Internal(format!("REST server: {e}")))
    };
    let grpc = grpc::serve_grpc(
        grpc_addr,
        state.clone(),
        state.limiter.clone(),
        Arc::clone(&state.metrics),
        api_key,
    );

    tokio::select! {
        r = rest => r?,
        r = grpc => r?,
    }
    Ok(())
}

fn parse_host(host: &str) -> Result<std::net::IpAddr> {
    host.parse()
        .map_err(|e| RetrievaError::Config(format!("invalid server.host '{host}': {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use retrieva_core::traits::Embedder;
    use retrieva_core::{Chunk, ChunkId, CollectionId, DocumentId};
    use std::collections::HashMap;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = Config {
            data_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let state = AppState::open_default(cfg).await.expect("state");
        let app = router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn metrics_exposes_prometheus_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = Config {
            data_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let state = AppState::open_default(cfg).await.expect("state");
        let app = router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_rejects_when_api_key_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cfg = Config {
            data_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        cfg.server.api_key = Some("secret".into());
        let state = AppState::open_default(cfg).await.expect("state");
        let app = router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"collection_id":"c","query":"q","top_k":1}"#))
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn search_bm25_after_manual_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = Config {
            data_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let state = AppState::open_default(cfg).await.expect("state");

        let chunk = Chunk {
            id: ChunkId::new("c1"),
            document_id: DocumentId::new("d1"),
            collection_id: CollectionId::new("demo"),
            text: "hybrid retrieval with bm25 and dense vectors".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        let emb = state
            .engine
            .embedder()
            .embed(&chunk.text)
            .await
            .expect("emb");
        state
            .index
            .upsert_chunks(
                &CollectionId::new("demo"),
                std::slice::from_ref(&chunk),
                Some(&[emb.values]),
            )
            .await
            .expect("upsert");

        let body = serde_json::json!({
            "collection_id": "demo",
            "query": "hybrid bm25",
            "top_k": 5,
            "mode": "bm25"
        });
        let app = router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn exact_cache_hit_on_second_search() {
        use retrieva_core::traits::Retriever;
        use retrieva_core::{SearchFilters, SearchMode, SearchRequest};

        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = Config {
            data_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let state = AppState::open_default(cfg).await.expect("state");
        let chunk = Chunk {
            id: ChunkId::new("c1"),
            document_id: DocumentId::new("d1"),
            collection_id: CollectionId::new("demo"),
            text: "cached hybrid retrieval query".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        let emb = state
            .engine
            .embedder()
            .embed(&chunk.text)
            .await
            .expect("emb");
        state
            .index
            .upsert_chunks(
                &CollectionId::new("demo"),
                std::slice::from_ref(&chunk),
                Some(&[emb.values]),
            )
            .await
            .expect("upsert");

        let req = SearchRequest {
            collection_id: CollectionId::new("demo"),
            query: "cached hybrid".into(),
            top_k: 5,
            mode: SearchMode::Bm25,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: true,
        };
        let first = state.engine.search(req.clone()).await.expect("first");
        assert!(!first.cache_hit);
        let second = state.engine.search(req).await.expect("second");
        assert!(second.cache_hit);
        assert_eq!(first.results.len(), second.results.len());
    }
}
