//! Configuration loaded from TOML files and `RETRIEVA_*` environment variables.
//!
//! Precedence (highest wins): environment → TOML file → built-in defaults.
//! Nested keys use `__` in env vars, e.g. `RETRIEVA_SERVER__PORT=8080`.

use crate::error::RetrievaError;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Error while loading or validating configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Figment failed to merge or deserialize.
    #[error("failed to load config: {0}")]
    Load(String),

    /// A required field failed validation.
    #[error("invalid config: {0}")]
    Invalid(String),
}

impl From<ConfigError> for RetrievaError {
    fn from(value: ConfigError) -> Self {
        RetrievaError::Config(value.to_string())
    }
}

/// Top-level Retrieva configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Data directory for indexes, WAL, and RocksDB.
    pub data_dir: PathBuf,
    /// Logging settings.
    pub logging: LoggingConfig,
    /// HTTP / gRPC server settings.
    pub server: ServerConfig,
    /// Retrieval defaults.
    pub search: SearchConfig,
    /// Document ingest / chunking settings.
    pub ingest: IngestConfig,
    /// Embedding model settings (used when `onnx` feature is enabled).
    pub embedding: EmbeddingConfig,
    /// Reranker settings.
    pub rerank: RerankConfig,
    /// Cache settings.
    pub cache: CacheConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            logging: LoggingConfig::default(),
            server: ServerConfig::default(),
            search: SearchConfig::default(),
            ingest: IngestConfig::default(),
            embedding: EmbeddingConfig::default(),
            rerank: RerankConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// `error` | `warn` | `info` | `debug` | `trace`
    pub level: String,
    /// `pretty` | `json`
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "pretty".into(),
        }
    }
}

/// Server bind addresses and auth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// REST listen host.
    pub host: String,
    /// REST listen port.
    pub port: u16,
    /// gRPC listen port.
    pub grpc_port: u16,
    /// Optional API key. When set, requests must present `Authorization: Bearer <key>`
    /// or `x-api-key: <key>`.
    pub api_key: Option<String>,
    /// Requests per second per API key (0 = unlimited).
    pub rate_limit_rps: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            grpc_port: 50051,
            api_key: None,
            rate_limit_rps: 0,
        }
    }
}

/// Document ingest and chunking parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Target chunk size in whitespace-separated words.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks, in words.
    pub chunk_overlap: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
        }
    }
}

/// Default search / fusion parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default `top_k`.
    pub default_top_k: usize,
    /// Candidates fetched from each retriever before fusion.
    pub candidate_pool: usize,
    /// Fusion strategy: `rrf` or `weighted`.
    pub fusion: String,
    /// RRF constant `k` (standard default is 60).
    pub rrf_k: u32,
    /// Weight for BM25 when using weighted fusion.
    pub bm25_weight: f32,
    /// Weight for dense when using weighted fusion.
    pub dense_weight: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_top_k: 10,
            candidate_pool: 100,
            fusion: "rrf".into(),
            rrf_k: 60,
            bm25_weight: 0.5,
            dense_weight: 0.5,
        }
    }
}

/// Embedding model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// ONNX model path (relative to cwd or absolute).
    pub model_path: PathBuf,
    /// HuggingFace / pinned model id for documentation and download scripts.
    pub model_id: String,
    /// Expected output dimensionality.
    pub dimensions: usize,
    /// Max batch size for ingest embedding.
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/bge-small-en-v1.5/model.onnx"),
            model_id: "BAAI/bge-small-en-v1.5".into(),
            dimensions: 384,
            batch_size: 32,
        }
    }
}

/// Reranker configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Active backend: `maxsim` (Phase 5 primary), `cross_encoder`, or `identity`.
    pub backend: String,
    /// Cross-encoder ONNX path.
    pub cross_encoder_path: PathBuf,
    /// Cross-encoder model id.
    pub cross_encoder_id: String,
    /// Late-interaction / ColBERT-style ONNX path (Phase 5).
    pub maxsim_path: PathBuf,
    /// Default candidates passed to the reranker.
    pub top_n: usize,
    /// Inference batch size.
    pub batch_size: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            backend: "maxsim".into(),
            cross_encoder_path: PathBuf::from("models/bge-reranker-base/model.onnx"),
            cross_encoder_id: "BAAI/bge-reranker-base".into(),
            maxsim_path: PathBuf::from("models/colbert/model.onnx"),
            top_n: 50,
            batch_size: 16,
        }
    }
}

/// Cache configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable exact-match query cache.
    pub exact_enabled: bool,
    /// Enable embedding-similarity semantic cache.
    pub semantic_enabled: bool,
    /// Max exact-cache entries.
    pub exact_capacity: u64,
    /// Cosine similarity threshold for semantic hits.
    pub semantic_threshold: f32,
    /// TTL in seconds (0 = no expiry).
    pub ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            exact_enabled: true,
            semantic_enabled: true,
            exact_capacity: 10_000,
            semantic_threshold: 0.97,
            ttl_secs: 3600,
        }
    }
}

impl Config {
    /// Load configuration: defaults ← optional TOML file ← `RETRIEVA_*` env.
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let mut figment = Figment::from(Serialized::defaults(Config::default()));

        if let Some(path) = path {
            if path.exists() {
                figment = figment.merge(Toml::file(path));
            } else {
                return Err(ConfigError::Load(format!(
                    "config file not found: {}",
                    path.display()
                )));
            }
        }

        // Nested keys: RETRIEVA_SERVER__PORT → server.port
        figment = figment.merge(Env::prefixed("RETRIEVA_").split("__"));

        let config: Config = figment
            .extract()
            .map_err(|e| ConfigError::Load(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Load from the default path `config/retrieva.toml` if it exists,
    /// otherwise defaults + env only.
    pub fn load_default() -> Result<Self, ConfigError> {
        let default_path = Path::new("config/retrieva.toml");
        if default_path.exists() {
            Self::load(Some(default_path))
        } else {
            Self::load(None)
        }
    }

    /// Load defaults ← TOML only (no env). Intended for unit tests.
    #[cfg(test)]
    fn load_toml_only(path: &Path) -> Result<Self, ConfigError> {
        let config: Config = Figment::from(Serialized::defaults(Config::default()))
            .merge(Toml::file(path))
            .extract()
            .map_err(|e| ConfigError::Load(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate cross-field invariants.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.server.port == 0 {
            return Err(ConfigError::Invalid("server.port must be non-zero".into()));
        }
        if self.server.grpc_port == 0 {
            return Err(ConfigError::Invalid(
                "server.grpc_port must be non-zero".into(),
            ));
        }
        if self.search.default_top_k == 0 {
            return Err(ConfigError::Invalid(
                "search.default_top_k must be >= 1".into(),
            ));
        }
        if self.ingest.chunk_size == 0 {
            return Err(ConfigError::Invalid(
                "ingest.chunk_size must be >= 1".into(),
            ));
        }
        if self.ingest.chunk_overlap >= self.ingest.chunk_size {
            return Err(ConfigError::Invalid(
                "ingest.chunk_overlap must be < ingest.chunk_size".into(),
            ));
        }
        if self.search.bm25_weight < 0.0 || self.search.dense_weight < 0.0 {
            return Err(ConfigError::Invalid(
                "search weights must be non-negative".into(),
            ));
        }
        match self.search.fusion.as_str() {
            "rrf" | "weighted" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "search.fusion must be 'rrf' or 'weighted', got '{other}'"
                )));
            }
        }
        match self.rerank.backend.as_str() {
            "identity" | "cross_encoder" | "maxsim" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "rerank.backend must be 'identity', 'cross_encoder', or 'maxsim', got '{other}'"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.cache.semantic_threshold) {
            return Err(ConfigError::Invalid(
                "cache.semantic_threshold must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn defaults_validate() {
        let cfg = Config::default();
        cfg.validate().expect("defaults must validate");
    }

    #[test]
    fn load_from_toml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("retrieva.toml");
        let mut file = fs::File::create(&path).expect("create");
        writeln!(
            file,
            r#"
data_dir = "/tmp/retrieva-data"

[server]
port = 9090

[search]
fusion = "weighted"
bm25_weight = 0.3
dense_weight = 0.7
"#
        )
        .expect("write");

        // Use TOML-only loader so ambient RETRIEVA_* env (e.g. from CLI smoke runs)
        // cannot flake this assertion.
        let cfg = Config::load_toml_only(&path).expect("load");
        assert_eq!(cfg.server.port, 9090);
        assert_eq!(cfg.search.fusion, "weighted");
        assert!((cfg.search.bm25_weight - 0.3).abs() < f32::EPSILON);
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/retrieva-data"));
    }

    #[test]
    fn missing_file_errors() {
        let err = Config::load(Some(Path::new("/nonexistent/retrieva.toml")));
        assert!(err.is_err());
    }

    #[test]
    fn invalid_fusion_rejected() {
        let mut cfg = Config::default();
        cfg.search.fusion = "magic".into();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_chunk_overlap_rejected() {
        let mut cfg = Config::default();
        cfg.ingest.chunk_overlap = cfg.ingest.chunk_size;
        assert!(cfg.validate().is_err());
    }
}
