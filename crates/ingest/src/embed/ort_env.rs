//! Process-wide ORT [`Environment`] (ort allows only one per process).

use ort::environment::Environment;
use ort::session::Session;
use retrieva_core::{Result, RetrievaError};
use std::path::Path;
use std::sync::OnceLock;

static ENV: OnceLock<Environment> = OnceLock::new();

/// Clone of the shared ONNX Runtime environment (created on first call).
pub fn shared_ort_env() -> Result<Environment> {
    if let Some(env) = ENV.get() {
        return Ok(env.clone());
    }
    let built = ort::init()
        .with_name("retrieva")
        .build()
        .map_err(|e| RetrievaError::Embedding(format!("ort env: {e}")))?;
    Ok(ENV.get_or_init(|| built).clone())
}

/// Build a session, preferring CUDA when the `cuda` feature is enabled.
pub fn ort_session_from_file(env: &Environment, model_path: &Path) -> Result<Session> {
    #[cfg(feature = "cuda")]
    {
        use tracing::{info, warn};
        match try_cuda_session(env, model_path) {
            Ok(session) => {
                info!(model = %model_path.display(), "ORT session using CUDA EP");
                return Ok(session);
            }
            Err(e) => {
                warn!(
                    error = %e,
                    model = %model_path.display(),
                    "CUDA EP unavailable; falling back to CPU"
                );
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = env;
    }
    Session::builder(env)
        .map_err(|e| RetrievaError::Embedding(format!("ort session builder: {e}")))?
        .commit_from_file(model_path)
        .map_err(|e| RetrievaError::Embedding(format!("ort load model: {e}")))
}

#[cfg(feature = "cuda")]
fn try_cuda_session(env: &Environment, model_path: &Path) -> Result<Session> {
    use ort::ep::CUDA;
    Session::builder(env)
        .map_err(|e| RetrievaError::Embedding(format!("ort session builder: {e}")))?
        .with_execution_providers([CUDA::default().build()])
        .map_err(|e| RetrievaError::Embedding(format!("ort CUDA EP: {e}")))?
        .commit_from_file(model_path)
        .map_err(|e| RetrievaError::Embedding(format!("ort load model (CUDA): {e}")))
}
