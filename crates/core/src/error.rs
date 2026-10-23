//! Typed errors for Retrieva library crates.

use thiserror::Error;

/// Convenient alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, RetrievaError>;

/// Top-level error type for Retrieva library code.
#[derive(Debug, Error)]
pub enum RetrievaError {
    /// Configuration failed to load or validate.
    #[error("config error: {0}")]
    Config(String),

    /// Requested collection was not found.
    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    /// Requested document was not found.
    #[error("document not found: {0}")]
    DocumentNotFound(String),

    /// Invalid argument supplied by the caller.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Embedding / ONNX inference failure.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Sparse (BM25) index failure.
    #[error("sparse index error: {0}")]
    SparseIndex(String),

    /// Dense (ANN) index failure.
    #[error("vector index error: {0}")]
    VectorIndex(String),

    /// Reranking failure.
    #[error("rerank error: {0}")]
    Rerank(String),

    /// Document / metadata store failure.
    #[error("storage error: {0}")]
    Storage(String),

    /// Cache failure.
    #[error("cache error: {0}")]
    Cache(String),

    /// Ingest / parsing failure.
    #[error("ingest error: {0}")]
    Ingest(String),

    /// Authentication / authorization failure.
    #[error("auth error: {0}")]
    Auth(String),

    /// Rate limit exceeded.
    #[error("rate limited")]
    RateLimited,

    /// Internal invariant violated.
    #[error("internal error: {0}")]
    Internal(String),

    /// I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl RetrievaError {
    /// Construct a config error from a displayable cause.
    pub fn config(err: impl ToString) -> Self {
        Self::Config(err.to_string())
    }

    /// Construct an internal error from a displayable cause.
    pub fn internal(err: impl ToString) -> Self {
        Self::Internal(err.to_string())
    }
}
