//! Shared domain types for retrieval requests and responses.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Opaque collection identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionId(pub String);

impl CollectionId {
    /// Create a collection id from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque document identifier (caller-supplied or generated).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    /// Create a document id from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque chunk identifier within a document.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    /// Create a chunk id from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A text chunk stored in the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Chunk id.
    pub id: ChunkId,
    /// Parent document id.
    pub document_id: DocumentId,
    /// Collection this chunk belongs to.
    pub collection_id: CollectionId,
    /// Chunk text.
    pub text: String,
    /// Zero-based ordinal within the parent document.
    pub ordinal: u32,
    /// Optional caller metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// A document to upsert (pre-chunking or already chunked).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Document id.
    pub id: DocumentId,
    /// Collection id.
    pub collection_id: CollectionId,
    /// Full document text (ingest will chunk if needed).
    pub text: String,
    /// Optional caller metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Search mode selecting which retrieval stages run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// BM25 / sparse only.
    Bm25,
    /// Dense ANN only.
    Dense,
    /// BM25 + dense with fusion.
    #[default]
    Hybrid,
    /// Hybrid candidates + reranker.
    #[serde(rename = "hybrid+rerank")]
    HybridRerank,
}

/// Per-stage score breakdown for a hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ScoreBreakdown {
    /// BM25 score, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<f32>,
    /// Dense similarity score, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense: Option<f32>,
    /// Fused score (RRF or weighted), if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused: Option<f32>,
    /// Reranker score, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f32>,
    /// Final ranking score returned to the client.
    pub final_score: f32,
}

/// Wall-clock latency per pipeline stage (milliseconds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LatencyBreakdown {
    /// Exact / semantic cache lookup.
    pub cache_ms: f64,
    /// Query embedding.
    pub embed_ms: f64,
    /// BM25 retrieval.
    pub bm25_ms: f64,
    /// Dense ANN retrieval.
    pub dense_ms: f64,
    /// Fusion of ranked lists.
    pub fusion_ms: f64,
    /// Reranking.
    pub rerank_ms: f64,
    /// End-to-end request latency.
    pub total_ms: f64,
}

impl LatencyBreakdown {
    /// Record a stage duration into the corresponding field.
    pub fn record(&mut self, stage: Stage, duration: Duration) {
        let ms = duration.as_secs_f64() * 1_000.0;
        match stage {
            Stage::Cache => self.cache_ms = ms,
            Stage::Embed => self.embed_ms = ms,
            Stage::Bm25 => self.bm25_ms = ms,
            Stage::Dense => self.dense_ms = ms,
            Stage::Fusion => self.fusion_ms = ms,
            Stage::Rerank => self.rerank_ms = ms,
            Stage::Total => self.total_ms = ms,
        }
    }
}

/// Named pipeline stage for latency accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Cache lookup.
    Cache,
    /// Embedding.
    Embed,
    /// BM25.
    Bm25,
    /// Dense ANN.
    Dense,
    /// Fusion.
    Fusion,
    /// Rerank.
    Rerank,
    /// Total.
    Total,
}

/// A scored retrieval hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredChunk {
    /// Retrieved chunk.
    pub chunk: Chunk,
    /// Score breakdown.
    pub scores: ScoreBreakdown,
}

/// Filter expression over document / chunk metadata (Phase 4 expands this).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    /// Exact-match metadata filters (`key` == `value`).
    #[serde(default)]
    pub metadata_eq: HashMap<String, String>,
}

/// Search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Target collection.
    pub collection_id: CollectionId,
    /// Query text.
    pub query: String,
    /// Number of final results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Retrieval mode.
    #[serde(default)]
    pub mode: SearchMode,
    /// Optional metadata filters.
    #[serde(default)]
    pub filters: SearchFilters,
    /// Candidates passed to the reranker (hybrid+rerank).
    #[serde(default = "default_rerank_top_n")]
    pub rerank_top_n: usize,
    /// Whether to consult the semantic / exact cache.
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

impl SearchRequest {
    /// Validate request parameters.
    pub fn validate(&self) -> crate::Result<()> {
        if self.query.trim().is_empty() {
            return Err(crate::RetrievaError::InvalidArgument(
                "query must not be empty".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(crate::RetrievaError::InvalidArgument(
                "top_k must be >= 1".into(),
            ));
        }
        if matches!(self.mode, SearchMode::HybridRerank) && self.rerank_top_n < self.top_k {
            return Err(crate::RetrievaError::InvalidArgument(
                "rerank_top_n must be >= top_k".into(),
            ));
        }
        Ok(())
    }
}

/// Search response with hits and per-stage latency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Ranked hits.
    pub results: Vec<ScoredChunk>,
    /// Per-stage latency breakdown.
    pub latency: LatencyBreakdown,
    /// Whether the response was served from cache.
    pub cache_hit: bool,
}

/// Hit/miss counters for a [`crate::traits::QueryCache`].
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Successful lookups.
    pub hits: u64,
    /// Lookups that missed.
    pub misses: u64,
}

impl CacheStats {
    /// `hits / (hits + misses)`, or `0.0` when no lookups.
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits.saturating_add(self.misses);
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

/// Dense embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Vector values (L2-normalized when produced by the default embedder).
    pub values: Vec<f32>,
}

impl Embedding {
    /// Create an embedding from values.
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// ANN neighbor hit (id + distance/similarity).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbor {
    /// Indexed vector key (typically chunk id encoded as u64).
    pub key: u64,
    /// Distance or similarity depending on metric.
    pub score: f32,
}

/// Collection metadata stored alongside indexes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection id.
    pub id: CollectionId,
    /// Human-readable name.
    pub name: String,
    /// Creation time (UTC).
    pub created_at: DateTime<Utc>,
    /// Number of indexed chunks.
    pub chunk_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_mode_serde_hybrid_rerank() {
        let mode = SearchMode::HybridRerank;
        let json = serde_json::to_string(&mode).expect("ser");
        assert_eq!(json, "\"hybrid+rerank\"");
        let back: SearchMode = serde_json::from_str(&json).expect("de");
        assert_eq!(back, SearchMode::HybridRerank);
    }

    #[test]
    fn validate_rejects_zero_top_k() {
        let req = SearchRequest {
            collection_id: CollectionId::new("c"),
            query: "q".into(),
            top_k: 0,
            mode: SearchMode::Bm25,
            filters: SearchFilters::default(),
            rerank_top_n: 50,
            use_cache: false,
        };
        assert!(req.validate().is_err());
    }
}
