//! Prometheus metrics registry and `/metrics` handler.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use parking_lot::Mutex;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use std::sync::Arc;
use std::time::Instant;

/// Shared Prometheus instruments.
#[derive(Clone)]
pub struct Metrics {
    registry: Registry,
    http_requests: IntCounterVec,
    grpc_requests: IntCounterVec,
    search_duration: HistogramVec,
    cache_hits: IntCounterVec,
    /// Guard so registry registration is once.
    _lock: Arc<Mutex<()>>,
}

impl Metrics {
    /// Register default Retrieva metrics.
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        let http_requests = IntCounterVec::new(
            Opts::new(
                "retrieva_http_requests_total",
                "HTTP requests by route and status",
            ),
            &["route", "status"],
        )?;
        let grpc_requests = IntCounterVec::new(
            Opts::new(
                "retrieva_grpc_requests_total",
                "gRPC requests by method and code",
            ),
            &["method", "code"],
        )?;
        let search_duration = HistogramVec::new(
            HistogramOpts::new(
                "retrieva_search_duration_seconds",
                "End-to-end search latency",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            ]),
            &["mode", "cache_hit"],
        )?;
        let cache_hits = IntCounterVec::new(
            Opts::new("retrieva_cache_lookups_total", "Cache lookup outcomes"),
            &["result"],
        )?;

        registry.register(Box::new(http_requests.clone()))?;
        registry.register(Box::new(grpc_requests.clone()))?;
        registry.register(Box::new(search_duration.clone()))?;
        registry.register(Box::new(cache_hits.clone()))?;

        Ok(Self {
            registry,
            http_requests,
            grpc_requests,
            search_duration,
            cache_hits,
            _lock: Arc::new(Mutex::new(())),
        })
    }

    /// Record an HTTP request.
    pub fn http(&self, route: &str, status: u16) {
        self.http_requests
            .with_label_values(&[route, &status.to_string()])
            .inc();
    }

    /// Record a gRPC request.
    pub fn grpc(&self, method: &str, code: &str) {
        self.grpc_requests.with_label_values(&[method, code]).inc();
    }

    /// Record search latency + cache outcome.
    pub fn search(&self, mode: &str, cache_hit: bool, started: Instant) {
        let hit = if cache_hit { "hit" } else { "miss" };
        self.search_duration
            .with_label_values(&[mode, hit])
            .observe(started.elapsed().as_secs_f64());
        self.cache_hits.with_label_values(&[hit]).inc();
    }

    /// Encode Prometheus text exposition format.
    pub fn gather(&self) -> Result<String, prometheus::Error> {
        let families = self.registry.gather();
        let encoder = TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&families, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new().expect("prometheus metrics")
    }
}

/// `GET /metrics` handler.
pub async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<crate::AppState>,
) -> Response {
    match state.metrics.gather() {
        Ok(body) => (
            StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            body,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}
