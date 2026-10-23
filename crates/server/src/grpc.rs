//! tonic gRPC service implementing `retrieva.v1.Retrieva`.

use crate::auth::grpc_auth_interceptor;
use crate::metrics::Metrics;
use crate::ratelimit::{check_limit, SharedLimiter};
use crate::AppState;
use retrieva_core::traits::{Embedder, Retriever};
use retrieva_core::{
    CollectionId, ScoreBreakdown, ScoredChunk, SearchFilters, SearchMode, SearchRequest,
    SearchResponse,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::info;

pub mod pb {
    //! Generated protobuf types (`tonic-prost-build`).
    #![allow(missing_docs)]
    include!(concat!(env!("OUT_DIR"), "/retrieva.v1.rs"));
}

use pb::retrieva_server::{Retrieva, RetrievaServer};
use pb::{
    HealthRequest, HealthResponse, IngestRequest, IngestResponse, LatencyBreakdown as PbLatency,
    ScoreBreakdown as PbScores, ScoredChunk as PbChunk, SearchRequest as PbSearchRequest,
    SearchResponse as PbSearchResponse,
};

struct RetrievaSvc {
    state: AppState,
    limiter: SharedLimiter,
    metrics: Arc<Metrics>,
}

#[tonic::async_trait]
impl Retrieva for RetrievaSvc {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.metrics.grpc("Health", "ok");
        Ok(Response::new(HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }

    async fn search(
        &self,
        request: Request<PbSearchRequest>,
    ) -> Result<Response<PbSearchResponse>, Status> {
        check_limit(&self.limiter).map_err(to_status)?;
        let started = Instant::now();
        let body = request.into_inner();
        let mode = parse_mode(&body.mode)?;
        let req = SearchRequest {
            collection_id: CollectionId::new(&body.collection_id),
            query: body.query,
            top_k: body.top_k.max(1) as usize,
            mode,
            filters: SearchFilters {
                metadata_eq: body.metadata_eq,
            },
            rerank_top_n: if body.rerank_top_n == 0 {
                50
            } else {
                body.rerank_top_n as usize
            },
            use_cache: body.use_cache,
        };
        let mode_label = format!("{:?}", req.mode);
        let resp = self.state.engine.search(req).await.map_err(to_status)?;
        self.metrics.search(&mode_label, resp.cache_hit, started);
        self.metrics.grpc("Search", "ok");
        Ok(Response::new(to_pb_search(resp)))
    }

    async fn ingest(
        &self,
        request: Request<IngestRequest>,
    ) -> Result<Response<IngestResponse>, Status> {
        check_limit(&self.limiter).map_err(to_status)?;
        let body = request.into_inner();
        let collection_id = CollectionId::new(&body.collection_id);
        let path = std::path::PathBuf::from(&body.path);
        let batch = retrieva_ingest::ingest_path(&path, &collection_id, &self.state.config.ingest)
            .map_err(to_status)?;

        self.state
            .index
            .upsert_documents(&batch.documents)
            .await
            .map_err(to_status)?;

        let texts: Vec<&str> = batch.chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self
            .state
            .engine
            .embedder()
            .embed_batch(&texts)
            .await
            .map_err(to_status)?;
        let vectors: Vec<Vec<f32>> = embeddings.into_iter().map(|e| e.values).collect();
        self.state
            .index
            .upsert_chunks(&collection_id, &batch.chunks, Some(&vectors))
            .await
            .map_err(to_status)?;
        self.state
            .cache
            .invalidate_collection(&collection_id)
            .await
            .map_err(to_status)?;

        self.metrics.grpc("Ingest", "ok");
        Ok(Response::new(IngestResponse {
            collection_id: body.collection_id,
            files_ok: batch.files_ok as u32,
            files_skipped: batch.files_skipped as u32,
            documents: batch.documents.len() as u32,
            chunks: batch.chunks.len() as u32,
        }))
    }
}

fn parse_mode(mode: &str) -> Result<SearchMode, Status> {
    if mode.is_empty() {
        return Ok(SearchMode::Hybrid);
    }
    match mode {
        "bm25" => Ok(SearchMode::Bm25),
        "dense" => Ok(SearchMode::Dense),
        "hybrid" => Ok(SearchMode::Hybrid),
        "hybrid+rerank" | "hybrid_rerank" => Ok(SearchMode::HybridRerank),
        other => Err(Status::invalid_argument(format!(
            "unknown mode '{other}' (expected bm25|dense|hybrid|hybrid+rerank)"
        ))),
    }
}

fn to_status(err: retrieva_core::RetrievaError) -> Status {
    use retrieva_core::RetrievaError::*;
    match err {
        InvalidArgument(m) => Status::invalid_argument(m),
        CollectionNotFound(m) | DocumentNotFound(m) => Status::not_found(m),
        Auth(m) => Status::unauthenticated(m),
        RateLimited => Status::resource_exhausted("rate limited"),
        other => Status::internal(other.to_string()),
    }
}

fn to_pb_search(resp: SearchResponse) -> PbSearchResponse {
    PbSearchResponse {
        results: resp.results.into_iter().map(to_pb_chunk).collect(),
        latency: Some(PbLatency {
            cache_ms: resp.latency.cache_ms,
            bm25_ms: resp.latency.bm25_ms,
            dense_ms: resp.latency.dense_ms,
            fusion_ms: resp.latency.fusion_ms,
            embed_ms: resp.latency.embed_ms,
            rerank_ms: resp.latency.rerank_ms,
            total_ms: resp.latency.total_ms,
        }),
        cache_hit: resp.cache_hit,
    }
}

fn to_pb_chunk(hit: ScoredChunk) -> PbChunk {
    PbChunk {
        chunk_id: hit.chunk.id.as_str().to_string(),
        document_id: hit.chunk.document_id.as_str().to_string(),
        collection_id: hit.chunk.collection_id.as_str().to_string(),
        text: hit.chunk.text,
        ordinal: hit.chunk.ordinal,
        metadata: hit.chunk.metadata,
        scores: Some(to_pb_scores(hit.scores)),
    }
}

fn to_pb_scores(s: ScoreBreakdown) -> PbScores {
    PbScores {
        bm25: s.bm25,
        dense: s.dense,
        fusion: s.fused,
        rerank: s.rerank,
        final_score: s.final_score,
    }
}

/// Serve gRPC on `addr` until cancelled.
pub async fn serve_grpc(
    addr: SocketAddr,
    state: AppState,
    limiter: SharedLimiter,
    metrics: Arc<Metrics>,
    api_key: Option<String>,
) -> Result<(), retrieva_core::RetrievaError> {
    let svc = RetrievaSvc {
        state,
        limiter,
        metrics,
    };
    let interceptor = grpc_auth_interceptor(api_key);
    // Health is authenticated too when api_key is set — clients can use insecure for probes
    // by leaving api_key unset. Documented in README.
    let server = RetrievaServer::with_interceptor(svc, interceptor);
    info!(%addr, "retrieva gRPC listening");
    tonic::transport::Server::builder()
        .add_service(server)
        .serve(addr)
        .await
        .map_err(|e| retrieva_core::RetrievaError::Internal(format!("grpc server: {e}")))
}
