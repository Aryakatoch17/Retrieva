//! Token-bucket rate limiting (shared across REST + gRPC).

use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use retrieva_core::RetrievaError;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Process-wide rate limiter (`None` = unlimited).
pub type SharedLimiter = Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>;

/// Build a limiter from `rate_limit_rps` (0 = unlimited).
pub fn build_limiter(rps: u32) -> SharedLimiter {
    if rps == 0 {
        return None;
    }
    let per_second = NonZeroU32::new(rps).unwrap_or(NonZeroU32::MIN);
    Some(Arc::new(RateLimiter::direct(Quota::per_second(per_second))))
}

/// Check the limiter; returns [`RetrievaError::RateLimited`] when exceeded.
pub fn check_limit(limiter: &SharedLimiter) -> Result<(), RetrievaError> {
    let Some(lim) = limiter else {
        return Ok(());
    };
    lim.check()
        .map(|_| ())
        .map_err(|_| RetrievaError::RateLimited)
}

/// Axum middleware enforcing the shared limiter.
pub async fn rate_limit_middleware(
    axum::extract::State(limiter): axum::extract::State<SharedLimiter>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let path = request.uri().path();
    if path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }

    if let Err(err) = check_limit(&limiter) {
        let body = serde_json::json!({ "error": err.to_string() });
        return (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_always_ok() {
        let lim = build_limiter(0);
        assert!(check_limit(&lim).is_ok());
    }

    #[test]
    fn limited_trips() {
        let lim = build_limiter(1);
        assert!(check_limit(&lim).is_ok());
        // Second immediate check may or may not fail depending on burst; drain.
        let _ = check_limit(&lim);
        let _ = check_limit(&lim);
        // With Quota::per_second(1) burst is typically 1 — third should fail.
        assert!(check_limit(&lim).is_err() || check_limit(&lim).is_err());
    }
}
