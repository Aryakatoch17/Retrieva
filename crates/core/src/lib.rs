//! Core types, traits, configuration, and errors for Retrieva.
//!
//! Library crates must not use `unwrap()` / `expect()` in non-test code.
//! Prefer [`RetrievaError`] for typed failures and `anyhow` only at binary edges.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod logging;
pub mod mocks;
pub mod traits;
pub mod types;

pub use config::{
    CacheConfig, Config, ConfigError, EmbeddingConfig, IngestConfig, RerankConfig, SearchConfig,
};
pub use error::{Result, RetrievaError};
pub use logging::{init_logging, LoggingOptions};
pub use types::*;
