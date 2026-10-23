//! Axum REST API (Phase 4+) with auth, rate limits, and metrics (Phase 8).

use crate::auth::require_api_key;
use crate::metrics::metrics_handler;
use crate::ratelimit::rate_limit_middleware;
use crate::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use retrieva_core::traits::{Embedder, Retriever};
use retrieva_core::{CollectionId, SearchFilters, SearchMode, SearchRequest, SearchResponse};
use retrieva_ingest::ingest_path;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use tower_http::trace::TraceLayer;

/// Build the REST router with auth / rate-limit layers from `state`.
pub fn router(state: AppState) -> Router {
    let api_key = state.config.server.api_key.clone();
    let limiter = state.limiter.clone();

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/v1/search", post(search))
        .route("/v1/ingest", post(ingest))
        .route(
            "/v1/collections/{collection}/search",
            post(search_collection),
        )
        .layer(middleware::from_fn_with_state(
            limiter,
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(api_key, require_api_key))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Debug, Deserialize)]
pub struct SearchBody {
    /// Target collection (also accepted as path param).
    #[serde(default)]
    pub collection_id: Option<String>,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub filters: SearchFilters,
    #[serde(default = "default_rerank_top_n")]
    pub rerank_top_n: usize,
    #[serde(default = "default_true")]
    pub use_cache: bool,
}

fn default_top_k() -> usize {
    10
}

fn default_rerank_top_n() -> usize {
    50
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct IngestBody {
    pub collection_id: String,
    /// Filesystem path (file or directory) relative to the server process.
    pub path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub collection_id: String,
    pub files_ok: usize,
    pub files_skipped: usize,
    pub documents: usize,
    pub chunks: usize,
}

async fn search(
    State(state): State<AppState>,
    Json(body): Json<SearchBody>,
) -> Result<Json<SearchResponse>, ApiError> {
    let collection = body
        .collection_id
        .clone()
        .ok_or_else(|| ApiError::bad_request("collection_id is required for POST /v1/search"))?;
    run_search(&state, &collection, body).await.map(Json)
}

async fn search_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Json(mut body): Json<SearchBody>,
) -> Result<Json<SearchResponse>, ApiError> {
    body.collection_id = Some(collection.clone());
    run_search(&state, &collection, body).await.map(Json)
}

async fn run_search(
    state: &AppState,
    collection: &str,
    body: SearchBody,
) -> Result<SearchResponse, ApiError> {
    let started = Instant::now();
    let mode_label = format!("{:?}", body.mode);
    let request = SearchRequest {
        collection_id: CollectionId::new(collection),
        query: body.query,
        top_k: body.top_k,
        mode: body.mode,
        filters: body.filters,
        rerank_top_n: body.rerank_top_n,
        use_cache: body.use_cache,
    };
    let resp = state
        .engine
        .search(request)
        .await
        .map_err(ApiError::from_retrieva)?;
    state.metrics.search(&mode_label, resp.cache_hit, started);
    state.metrics.http("/v1/search", 200);
    Ok(resp)
}

async fn ingest(
    State(state): State<AppState>,
    Json(body): Json<IngestBody>,
) -> Result<Json<IngestResponse>, ApiError> {
    let collection_id = CollectionId::new(&body.collection_id);
    let batch = ingest_path(&body.path, &collection_id, &state.config.ingest)
        .map_err(ApiError::from_retrieva)?;

    state
        .index
        .upsert_documents(&batch.documents)
        .await
        .map_err(ApiError::from_retrieva)?;

    let texts: Vec<&str> = batch.chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = state
        .engine
        .embedder()
        .embed_batch(&texts)
        .await
        .map_err(ApiError::from_retrieva)?;
    let vectors: Vec<Vec<f32>> = embeddings.into_iter().map(|e| e.values).collect();
    state
        .index
        .upsert_chunks(&collection_id, &batch.chunks, Some(&vectors))
        .await
        .map_err(ApiError::from_retrieva)?;

    state
        .cache
        .invalidate_collection(&collection_id)
        .await
        .map_err(ApiError::from_retrieva)?;

    state.metrics.http("/v1/ingest", 200);
    Ok(Json(IngestResponse {
        collection_id: body.collection_id,
        files_ok: batch.files_ok,
        files_skipped: batch.files_skipped,
        documents: batch.documents.len(),
        chunks: batch.chunks.len(),
    }))
}

/// HTTP error wrapper.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn from_retrieva(err: retrieva_core::RetrievaError) -> Self {
        use retrieva_core::RetrievaError::*;
        let status = match &err {
            InvalidArgument(_) => StatusCode::BAD_REQUEST,
            CollectionNotFound(_) | DocumentNotFound(_) => StatusCode::NOT_FOUND,
            Auth(_) => StatusCode::UNAUTHORIZED,
            RateLimited => StatusCode::TOO_MANY_REQUESTS,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}
