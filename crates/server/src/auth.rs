//! API-key authentication for REST and gRPC.

use axum::extract::Request;
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use retrieva_core::RetrievaError;
use tonic::metadata::MetadataMap;
use tonic::Status;

/// Extract a presented API key from HTTP headers.
pub fn extract_http_api_key(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        if let Some(rest) = v
            .strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
        {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Extract a presented API key from gRPC metadata.
pub fn extract_grpc_api_key(meta: &MetadataMap) -> Option<String> {
    if let Some(v) = meta.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    if let Some(v) = meta.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(rest) = v
            .strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
        {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Validate key against configured expectation (`None` = auth disabled).
pub fn check_api_key(expected: Option<&str>, presented: Option<&str>) -> Result<(), RetrievaError> {
    let Some(expected) = expected.filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    match presented {
        Some(got) if got == expected => Ok(()),
        Some(_) => Err(RetrievaError::Auth("invalid API key".into())),
        None => Err(RetrievaError::Auth("missing API key".into())),
    }
}

/// Axum middleware: enforce `server.api_key` when configured.
pub async fn require_api_key(
    axum::extract::State(expected): axum::extract::State<Option<String>>,
    request: Request,
    next: Next,
) -> Response {
    // Public endpoints.
    let path = request.uri().path();
    if path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }

    let presented = extract_http_api_key(request.headers());
    if let Err(err) = check_api_key(expected.as_deref(), presented.as_deref()) {
        let status = StatusCode::UNAUTHORIZED;
        let body = serde_json::json!({ "error": err.to_string() });
        return (status, axum::Json(body)).into_response();
    }
    next.run(request).await
}

/// gRPC interceptor factory.
pub fn grpc_auth_interceptor(
    expected: Option<String>,
) -> impl FnMut(tonic::Request<()>) -> Result<tonic::Request<()>, Status> + Clone {
    move |req: tonic::Request<()>| {
        let presented = extract_grpc_api_key(req.metadata());
        check_api_key(expected.as_deref(), presented.as_deref())
            .map(|_| req)
            .map_err(|e| match e {
                RetrievaError::Auth(msg) => Status::unauthenticated(msg),
                other => Status::internal(other.to_string()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    #[test]
    fn accepts_bearer_and_x_api_key() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", "secret".parse().unwrap());
        assert_eq!(extract_http_api_key(&h).as_deref(), Some("secret"));

        let mut h2 = HeaderMap::new();
        h2.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        assert_eq!(extract_http_api_key(&h2).as_deref(), Some("secret"));
    }

    #[test]
    fn check_disabled_when_unset() {
        assert!(check_api_key(None, None).is_ok());
        assert!(check_api_key(Some(""), None).is_ok());
    }

    #[test]
    fn check_rejects_missing_or_wrong() {
        assert!(check_api_key(Some("k"), None).is_err());
        assert!(check_api_key(Some("k"), Some("nope")).is_err());
        assert!(check_api_key(Some("k"), Some("k")).is_ok());
    }
}
