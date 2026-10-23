//! ONNX late-interaction encoder producing token embeddings for MaxSim.

use crate::maxsim::maxsim_score;
use async_trait::async_trait;
use dashmap::DashMap;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use retrieva_core::traits::Reranker;
use retrieva_core::{RerankConfig, Result, RetrievaError, ScoredChunk};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// ColBERT-style MaxSim using an ONNX model that emits `last_hidden_state`.
///
/// Document token vectors are cached by chunk id so search-time work is query
/// encode + MaxSim only (after [`Self::precompute_chunks`] or first sighting).
pub struct OnnxMaxSimReranker {
    _env: Environment,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    /// Vespa ColBERT exports omit `token_type_ids`.
    wants_token_type_ids: bool,
    /// Precomputed / lazily filled doc token embeddings keyed by chunk id.
    doc_tokens: DashMap<String, Arc<Vec<Vec<f32>>>>,
}

impl OnnxMaxSimReranker {
    /// Load from [`RerankConfig::maxsim_path`] (+ sibling `tokenizer.json`).
    pub fn from_config(cfg: &RerankConfig) -> Result<Self> {
        let model = &cfg.maxsim_path;
        let parent = model.parent().unwrap_or_else(|| Path::new("."));
        Self::open(model, &parent.join("tokenizer.json"))
    }

    /// Open model + tokenizer.
    pub fn open(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(RetrievaError::Rerank(format!(
                "MaxSim ONNX not found at {} (see scripts/download_colbert_hint.md)",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(RetrievaError::Rerank(format!(
                "tokenizer.json not found at {}",
                tokenizer_path.display()
            )));
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RetrievaError::Rerank(format!("tokenizer: {e}")))?;
        let env = retrieva_ingest::shared_ort_env()
            .map_err(|e| RetrievaError::Rerank(format!("ort env: {e}")))?;
        let session = retrieva_ingest::ort_session_from_file(&env, model_path)
            .map_err(|e| RetrievaError::Rerank(format!("ort load: {e}")))?;
        let wants_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        debug!(
            model = %model_path.display(),
            wants_token_type_ids,
            "loaded ONNX MaxSim encoder"
        );

        Ok(Self {
            _env: env,
            session: Mutex::new(session),
            tokenizer,
            max_length: 180,
            wants_token_type_ids,
            doc_tokens: DashMap::new(),
        })
    }

    /// Encode and store document token vectors (call at index time).
    pub fn precompute_chunks<'a, I>(&self, chunks: I) -> Result<usize>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut n = 0usize;
        for (id, text) in chunks {
            if self.doc_tokens.contains_key(id) {
                continue;
            }
            let toks = self.encode_tokens(text)?;
            self.doc_tokens.insert(id.to_string(), Arc::new(toks));
            n += 1;
        }
        info!(precomputed = n, cached = self.doc_tokens.len(), "MaxSim doc tokens");
        Ok(n)
    }

    /// Number of cached document encodings.
    pub fn cached_docs(&self) -> usize {
        self.doc_tokens.len()
    }

    fn doc_tokens_for(&self, chunk_id: &str, text: &str) -> Result<Arc<Vec<Vec<f32>>>> {
        if let Some(hit) = self.doc_tokens.get(chunk_id) {
            return Ok(Arc::clone(hit.value()));
        }
        let toks = Arc::new(self.encode_tokens(text)?);
        self.doc_tokens
            .insert(chunk_id.to_string(), Arc::clone(&toks));
        Ok(toks)
    }

    fn encode_tokens(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RetrievaError::Rerank(format!("tokenize: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let types = encoding.get_type_ids();
        let n = ids.len().min(self.max_length).max(1);

        let mut input_ids = vec![0i64; n];
        let mut attention_mask = vec![0i64; n];
        let mut token_type_ids = vec![0i64; n];
        for s in 0..n {
            input_ids[s] = i64::from(ids[s]);
            attention_mask[s] = i64::from(mask.get(s).copied().unwrap_or(1));
            token_type_ids[s] = i64::from(types.get(s).copied().unwrap_or(0));
        }

        let mut session = self.session.lock();
        let outputs = if self.wants_token_type_ids {
            session
                .run(ort::inputs![
                    "input_ids" => map_ort(Tensor::from_array(([1usize, n], input_ids)))?,
                    "attention_mask" => map_ort(Tensor::from_array(([1usize, n], attention_mask.clone())))?,
                    "token_type_ids" => map_ort(Tensor::from_array(([1usize, n], token_type_ids)))?,
                ])
                .map_err(|e| RetrievaError::Rerank(format!("ort run: {e}")))?
        } else {
            let _ = token_type_ids;
            session
                .run(ort::inputs![
                    "input_ids" => map_ort(Tensor::from_array(([1usize, n], input_ids)))?,
                    "attention_mask" => map_ort(Tensor::from_array(([1usize, n], attention_mask.clone())))?,
                ])
                .map_err(|e| RetrievaError::Rerank(format!("ort run: {e}")))?
        };

        // Prefer Vespa `contextual`, then HF `last_hidden_state`, else first output.
        let (dims, flat) = if let Some(v) = outputs
            .get("contextual")
            .or_else(|| outputs.get("last_hidden_state"))
        {
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| RetrievaError::Rerank(format!("extract: {e}")))?;
            (
                shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                data.to_vec(),
            )
        } else {
            let v = outputs
                .values()
                .next()
                .ok_or_else(|| RetrievaError::Rerank("no ONNX outputs".into()))?;
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| RetrievaError::Rerank(format!("extract: {e}")))?;
            (
                shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                data.to_vec(),
            )
        };
        if dims.len() != 3 {
            return Err(RetrievaError::Rerank(format!(
                "expected [1,seq,dim] hidden states, got {dims:?}"
            )));
        }
        let (_b, seq, hidden) = (dims[0], dims[1], dims[2]);
        let mut tokens = Vec::new();
        for s in 0..seq.min(n) {
            if attention_mask[s] == 0 {
                continue;
            }
            let base = s * hidden;
            let mut vec = flat[base..base + hidden].to_vec();
            l2_normalize(&mut vec);
            tokens.push(vec);
        }
        Ok(tokens)
    }
}

fn map_ort<T>(r: std::result::Result<T, ort::Error>) -> Result<T> {
    r.map_err(|e| RetrievaError::Rerank(e.to_string()))
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in values {
        *v /= norm;
    }
}

#[async_trait]
impl Reranker for OnnxMaxSimReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[ScoredChunk],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        let q_toks = self.encode_tokens(query)?;
        let mut out = Vec::with_capacity(candidates.len());
        for hit in candidates {
            let d_toks = self.doc_tokens_for(hit.chunk.id.as_str(), &hit.chunk.text)?;
            let score = maxsim_score(&q_toks, d_toks.as_ref());
            let mut hit = hit.clone();
            hit.scores.rerank = Some(score);
            hit.scores.final_score = score;
            out.push(hit);
        }
        out.sort_by(|a, b| {
            b.scores
                .final_score
                .partial_cmp(&a.scores.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(top_k);
        Ok(out)
    }
}
