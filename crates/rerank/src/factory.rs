//! Build a [`Reranker`] from [`RerankConfig`].

use crate::identity::IdentityReranker;
use crate::maxsim::HashMaxSimReranker;
use retrieva_core::traits::Reranker;
use retrieva_core::{RerankConfig, Result, RetrievaError};
use std::sync::Arc;

/// Construct the configured reranker backend.
///
/// | backend | default build | `--features onnx` |
/// |---------|---------------|-------------------|
/// | `identity` | [`IdentityReranker`] | same |
/// | `maxsim` | [`HashMaxSimReranker`] | [`OnnxMaxSimReranker`] if model present, else hash |
/// | `cross_encoder` | error unless onnx | [`CrossEncoderReranker`] |
pub fn reranker_from_config(cfg: &RerankConfig) -> Result<Arc<dyn Reranker>> {
    match cfg.backend.as_str() {
        "identity" => Ok(Arc::new(IdentityReranker)),
        "maxsim" => maxsim_backend(cfg),
        "cross_encoder" => cross_encoder_backend(cfg),
        other => Err(RetrievaError::InvalidArgument(format!(
            "unknown rerank.backend '{other}' (expected identity|maxsim|cross_encoder)"
        ))),
    }
}

fn maxsim_backend(cfg: &RerankConfig) -> Result<Arc<dyn Reranker>> {
    #[cfg(feature = "onnx")]
    {
        if cfg.maxsim_path.exists() {
            return Ok(Arc::new(
                crate::maxsim_onnx::OnnxMaxSimReranker::from_config(cfg)?,
            ));
        }
        tracing::warn!(
            path = %cfg.maxsim_path.display(),
            "maxsim ONNX missing; falling back to HashMaxSimReranker"
        );
    }
    let _ = cfg;
    Ok(Arc::new(HashMaxSimReranker::new(32)))
}

fn cross_encoder_backend(cfg: &RerankConfig) -> Result<Arc<dyn Reranker>> {
    #[cfg(feature = "onnx")]
    {
        return Ok(Arc::new(
            crate::cross_encoder::CrossEncoderReranker::from_config(cfg)?,
        ));
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = cfg;
        Err(RetrievaError::Rerank(
            "cross_encoder requires `--features onnx` and model files".into(),
        ))
    }
}
