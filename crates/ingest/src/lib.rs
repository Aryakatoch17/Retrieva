//! Document ingestion, chunking, and batched embedding.
//!
//! Phase 2: parsers + word chunker. Phase 3: [`embed`] (hash always; ONNX behind `onnx`).

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod chunk;
pub mod embed;
pub mod parse;
pub mod pipeline;

pub use chunk::chunk_text;
pub use embed::HashEmbedder;
#[cfg(feature = "onnx")]
pub use embed::{ort_session_from_file, shared_ort_env, OnnxEmbedder, BGE_QUERY_PREFIX};
pub use parse::{extract_text, is_supported};
pub use pipeline::{ingest_file, ingest_path, IngestBatch};
