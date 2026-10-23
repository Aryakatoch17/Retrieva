//! ONNX cross-encoder reranker (`BAAI/bge-reranker-base` style).

use async_trait::async_trait;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use retrieva_core::traits::Reranker;
use retrieva_core::{RerankConfig, Result, RetrievaError, ScoredChunk};
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::debug;

/// Cross-encoder that scores `(query, passage)` pairs via ONNX.
pub struct CrossEncoderReranker {
    _env: Environment,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    batch_size: usize,
}

impl CrossEncoderReranker {
    /// Load from [`RerankConfig`] (`cross_encoder_path` + sibling `tokenizer.json`).
    pub fn from_config(cfg: &RerankConfig) -> Result<Self> {
        let model = &cfg.cross_encoder_path;
        let parent = model.parent().unwrap_or_else(|| Path::new("."));
        Self::open(model, &parent.join("tokenizer.json"), cfg.batch_size)
    }

    /// Open model + tokenizer from explicit paths.
    pub fn open(model_path: &Path, tokenizer_path: &Path, batch_size: usize) -> Result<Self> {
        if !model_path.exists() {
            return Err(RetrievaError::Rerank(format!(
                "cross-encoder ONNX not found at {} \
                 (download with scripts/download_bge_reranker.py)",
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
        // BGE-reranker (XLM-R) exports only `input_ids` + `attention_mask`.
        // Do not feed `token_type_ids` even if a tokenizer produces type ids.
        let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        debug!(
            model = %model_path.display(),
            ?input_names,
            "loaded cross-encoder"
        );

        Ok(Self {
            _env: env,
            session: Mutex::new(session),
            tokenizer,
            max_length: 512,
            batch_size: batch_size.max(1),
        })
    }

    fn score_pairs(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(passages.len());
        for chunk in passages.chunks(self.batch_size) {
            scores.extend(self.score_batch(query, chunk)?);
        }
        Ok(scores)
    }

    fn score_batch(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let pairs: Vec<(String, String)> = passages
            .iter()
            .map(|p| (query.to_string(), (*p).to_string()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| RetrievaError::Rerank(format!("tokenize: {e}")))?;

        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.len().min(self.max_length))
            .max()
            .unwrap_or(1)
            .max(1);

        let mut input_ids = vec![0i64; batch * seq];
        let mut attention_mask = vec![0i64; batch * seq];
        for (b, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let n = ids.len().min(seq);
            for s in 0..n {
                let i = b * seq + s;
                input_ids[i] = i64::from(ids[s]);
                attention_mask[i] = i64::from(mask.get(s).copied().unwrap_or(1));
            }
        }

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => map_ort(Tensor::from_array(([batch, seq], input_ids)))?,
                "attention_mask" => map_ort(Tensor::from_array(([batch, seq], attention_mask)))?,
            ])
            .map_err(|e| RetrievaError::Rerank(format!("ort run: {e}")))?;

        let first = outputs
            .values()
            .next()
            .ok_or_else(|| RetrievaError::Rerank("no ONNX outputs".into()))?;
        let (_shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| RetrievaError::Rerank(format!("extract: {e}")))?;

        // logits shape [batch] or [batch, 1]
        if data.len() == batch {
            Ok(data.to_vec())
        } else if data.len() == batch * 1 {
            Ok(data.to_vec())
        } else if data.len() >= batch {
            Ok(data
                .iter()
                .step_by(data.len() / batch)
                .copied()
                .take(batch)
                .collect())
        } else {
            Err(RetrievaError::Rerank(format!(
                "unexpected logits len {} for batch {batch}",
                data.len()
            )))
        }
    }
}

fn map_ort<T>(r: std::result::Result<T, ort::Error>) -> Result<T> {
    r.map_err(|e| RetrievaError::Rerank(e.to_string()))
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[ScoredChunk],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let passages: Vec<&str> = candidates.iter().map(|c| c.chunk.text.as_str()).collect();
        let scores = self.score_pairs(query, &passages)?;

        let mut out: Vec<ScoredChunk> = candidates
            .iter()
            .zip(scores)
            .map(|(hit, score)| {
                let mut hit = hit.clone();
                hit.scores.rerank = Some(score);
                hit.scores.final_score = score;
                hit
            })
            .collect();
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
