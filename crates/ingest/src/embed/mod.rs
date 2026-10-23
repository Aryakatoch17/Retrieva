//! Text embedders: always-on hash baseline + optional ONNX (`bge-small-en-v1.5`).

mod hash;
#[cfg(feature = "onnx")]
mod onnx;
#[cfg(feature = "onnx")]
mod ort_env;

pub use hash::HashEmbedder;
#[cfg(feature = "onnx")]
pub use onnx::{OnnxEmbedder, BGE_QUERY_PREFIX};
#[cfg(feature = "onnx")]
pub use ort_env::{ort_session_from_file, shared_ort_env};
