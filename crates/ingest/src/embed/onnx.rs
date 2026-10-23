//! ONNX Runtime embedder for `BAAI/bge-small-en-v1.5` (mean-pool + L2).

use super::ort_env::shared_ort_env;
use async_trait::async_trait;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use retrieva_core::traits::Embedder;
use retrieva_core::{Embedding, EmbeddingConfig, Result, RetrievaError};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::{Encoding, Tokenizer};
use tracing::debug;

/// Default query instruction used by BGE retrieval models.
pub const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching: ";

/// ONNX embedder matching the Sentence-Transformers / BGE pooling recipe.
pub struct OnnxEmbedder {
    _env: Environment,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimensions: usize,
    max_length: usize,
    batch_size: usize,
    /// When true, [`Embedder::embed`] / `embed_batch` apply [`BGE_QUERY_PREFIX`].
    query_mode: AtomicBool,
}

impl OnnxEmbedder {
    /// Load tokenizer + ONNX model from `config.model_path`'s parent directory.
    ///
    /// Expects:
    /// - `{parent}/model.onnx` (or the configured `model_path`)
    /// - `{parent}/tokenizer.json`
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
        let model_path = &config.model_path;
        let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
        let tokenizer_path = parent.join("tokenizer.json");
        Self::open(
            model_path,
            &tokenizer_path,
            config.dimensions,
            config.batch_size,
            false,
        )
    }

    /// Open an embedder from explicit paths.
    pub fn open(
        model_path: &Path,
        tokenizer_path: &Path,
        dimensions: usize,
        batch_size: usize,
        query_mode: bool,
    ) -> Result<Self> {
        if !model_path.exists() {
            return Err(RetrievaError::Embedding(format!(
                "ONNX model not found at {} (download with scripts/download_bge_small.py)",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(RetrievaError::Embedding(format!(
                "tokenizer.json not found at {}",
                tokenizer_path.display()
            )));
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RetrievaError::Embedding(format!("failed to load tokenizer: {e}")))?;

        let env = shared_ort_env()?;
        let session = super::ort_env::ort_session_from_file(&env, model_path)?;

        debug!(
            model = %model_path.display(),
            dimensions,
            query_mode,
            "loaded ONNX embedder"
        );

        Ok(Self {
            _env: env,
            session: Mutex::new(session),
            tokenizer,
            dimensions,
            max_length: 512,
            batch_size: batch_size.max(1),
            query_mode: AtomicBool::new(query_mode),
        })
    }

    /// Builder-style: embed texts as queries (with BGE prefix).
    pub fn with_query_mode(self, query_mode: bool) -> Self {
        self.query_mode.store(query_mode, Ordering::Relaxed);
        self
    }

    /// Flip query prefix without rebuilding the ONNX session (same process-wide ORT env).
    pub fn set_query_mode(&self, query_mode: bool) {
        self.query_mode.store(query_mode, Ordering::Relaxed);
    }

    /// Embed a retrieval query with the BGE instruction prefix (ignores `query_mode`).
    pub fn embed_query_sync(&self, text: &str) -> Result<Embedding> {
        let prefixed = format!("{BGE_QUERY_PREFIX}{text}");
        self.embed_raw(&[prefixed.as_str()])?
            .into_iter()
            .next()
            .ok_or_else(|| RetrievaError::Embedding("empty embedder output".into()))
    }

    /// Embed document passages with no query prefix (ignores `query_mode`).
    pub fn embed_documents_sync(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        self.embed_raw(texts)
    }

    fn prepare_texts(&self, texts: &[&str]) -> Vec<String> {
        if self.query_mode.load(Ordering::Relaxed) {
            texts
                .iter()
                .map(|t| format!("{BGE_QUERY_PREFIX}{t}"))
                .collect()
        } else {
            texts.iter().map(|t| (*t).to_string()).collect()
        }
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = self.prepare_texts(texts);
        let refs: Vec<&str> = prepared.iter().map(String::as_str).collect();
        self.embed_raw(&refs)
    }

    fn embed_raw(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            out.extend(self.embed_batch_inner(chunk)?);
        }
        Ok(out)
    }

    fn embed_batch_inner(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RetrievaError::Embedding(format!("tokenize: {e}")))?;

        let (batch, seq, input_ids, attention_mask, token_type_ids) =
            encodings_to_flat(&encodings, self.max_length);

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => map_ort(Tensor::from_array(([batch, seq], input_ids)))?,
                "attention_mask" => map_ort(Tensor::from_array(([batch, seq], attention_mask.clone())))?,
                "token_type_ids" => map_ort(Tensor::from_array(([batch, seq], token_type_ids)))?,
            ])
            .map_err(|e| RetrievaError::Embedding(format!("ort run: {e}")))?;

        let (dims, flat) = if let Some(v) = outputs.get("last_hidden_state") {
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| RetrievaError::Embedding(format!("extract tensor: {e}")))?;
            (
                shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                data.to_vec(),
            )
        } else {
            let v = outputs
                .values()
                .next()
                .ok_or_else(|| RetrievaError::Embedding("no ONNX outputs".into()))?;
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| RetrievaError::Embedding(format!("extract tensor: {e}")))?;
            (
                shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                data.to_vec(),
            )
        };
        if dims.len() != 3 {
            return Err(RetrievaError::Embedding(format!(
                "expected rank-3 hidden states, got shape {dims:?}"
            )));
        }
        let (batch_out, seq_out, hidden) = (dims[0], dims[1], dims[2]);
        if hidden != self.dimensions {
            debug!(
                configured = self.dimensions,
                actual = hidden,
                "embedding dim differs from config; using model hidden size"
            );
        }
        let mut embeddings = Vec::with_capacity(batch_out);
        for b in 0..batch_out {
            let mut acc = vec![0.0_f32; hidden];
            let mut count = 0.0_f32;
            for s in 0..seq_out {
                if attention_mask[b * seq + s] == 0 {
                    continue;
                }
                count += 1.0;
                let base = (b * seq_out + s) * hidden;
                for h in 0..hidden {
                    acc[h] += flat[base + h];
                }
            }
            let denom = count.max(1e-9);
            for v in &mut acc {
                *v /= denom;
            }
            l2_normalize(&mut acc);
            embeddings.push(Embedding::new(acc));
        }
        Ok(embeddings)
    }
}

fn map_ort<T>(r: std::result::Result<T, ort::Error>) -> Result<T> {
    r.map_err(|e| RetrievaError::Embedding(e.to_string()))
}

fn encodings_to_flat(
    encodings: &[Encoding],
    max_length: usize,
) -> (usize, usize, Vec<i64>, Vec<i64>, Vec<i64>) {
    let batch = encodings.len();
    let seq = encodings
        .iter()
        .map(|e| e.len().min(max_length))
        .max()
        .unwrap_or(0)
        .max(1);

    let mut input_ids = vec![0i64; batch * seq];
    let mut attention_mask = vec![0i64; batch * seq];
    let mut token_type_ids = vec![0i64; batch * seq];

    for (b, enc) in encodings.iter().enumerate() {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let types = enc.get_type_ids();
        let n = ids.len().min(seq);
        for s in 0..n {
            let i = b * seq + s;
            input_ids[i] = i64::from(ids[s]);
            attention_mask[i] = i64::from(mask.get(s).copied().unwrap_or(1));
            token_type_ids[i] = i64::from(types.get(s).copied().unwrap_or(0));
        }
    }
    (batch, seq, input_ids, attention_mask, token_type_ids)
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in values {
        *v /= norm;
    }
}

#[async_trait]
impl Embedder for OnnxEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Embedding> {
        self.embed_batch_sync(&[text])?
            .into_iter()
            .next()
            .ok_or_else(|| RetrievaError::Embedding("empty embedder output".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        self.embed_batch_sync(texts)
    }
}
