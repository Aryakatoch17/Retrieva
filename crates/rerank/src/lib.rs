//! Rerankers for Retrieva.
//!
//! Phase 4: [`IdentityReranker`] + optional ONNX cross-encoder.
//! Phase 5: **MaxSim** primary ([`HashMaxSimReranker`] / ONNX) with ablation vs cross-encoder.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ablate;
#[cfg(feature = "onnx")]
mod cross_encoder;
mod factory;
mod identity;
mod maxsim;
#[cfg(feature = "onnx")]
mod maxsim_onnx;

pub use ablate::{ablate_rerankers, top1_agreement, AblationArm, AblationReport};
#[cfg(feature = "onnx")]
pub use cross_encoder::CrossEncoderReranker;
pub use factory::reranker_from_config;
pub use identity::IdentityReranker;
pub use maxsim::{hash_token_embedding, maxsim_score, tokenize, HashMaxSimReranker};
#[cfg(feature = "onnx")]
pub use maxsim_onnx::OnnxMaxSimReranker;
