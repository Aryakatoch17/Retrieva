//! ColBERT-style MaxSim scoring and a deterministic hash-token MaxSim reranker.

use async_trait::async_trait;
use retrieva_core::traits::Reranker;
use retrieva_core::{Result, ScoredChunk};

/// Compute MaxSim: `Σ_i max_j cos(q_i, d_j)`.
pub fn maxsim_score(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| cosine(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0)
        })
        .sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Whitespace tokenization (lowercased); good enough for the hash MaxSim baseline.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Deterministic per-token embedding (L2-normalized).
///
/// Uses a stable byte mix (not [`std::collections::hash_map::DefaultHasher`],
/// which is keyed randomly per process and is a poor embedding basis).
pub fn hash_token_embedding(token: &str, dims: usize) -> Vec<f32> {
    let dims = dims.max(1);
    let mut values = vec![0.0_f32; dims];
    for (i, byte) in token.bytes().enumerate() {
        let idx = i % dims;
        values[idx] += f32::from(byte) / 255.0;
        // Mix neighboring dims so short tokens still spread.
        values[(idx + 1) % dims] += f32::from(byte.rotate_left(3)) / 255.0;
    }
    // Salt by length so anagrams differ slightly.
    let salt = token.len() as f32 * 0.017;
    for (i, v) in values.iter_mut().enumerate() {
        *v += salt * ((i as f32 + 1.0).sin());
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in &mut values {
        *v /= norm;
    }
    values
}

fn embed_tokens(text: &str, dims: usize) -> Vec<Vec<f32>> {
    tokenize(text)
        .into_iter()
        .map(|t| hash_token_embedding(&t, dims))
        .collect()
}

/// Late-interaction MaxSim reranker using hash token embeddings (no ONNX).
///
/// Useful as the Phase 5 default path and for ablation without model weights.
#[derive(Debug, Clone)]
pub struct HashMaxSimReranker {
    /// Token embedding dimensionality.
    pub dims: usize,
}

impl Default for HashMaxSimReranker {
    fn default() -> Self {
        Self { dims: 32 }
    }
}

impl HashMaxSimReranker {
    /// Create a hash MaxSim reranker.
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(1) }
    }
}

#[async_trait]
impl Reranker for HashMaxSimReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[ScoredChunk],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        let q_toks = embed_tokens(query, self.dims);
        let mut out: Vec<ScoredChunk> = candidates
            .iter()
            .map(|hit| {
                let d_toks = embed_tokens(&hit.chunk.text, self.dims);
                let score = maxsim_score(&q_toks, &d_toks);
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

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{Chunk, ChunkId, CollectionId, DocumentId, ScoreBreakdown};
    use std::collections::HashMap;

    #[test]
    fn identical_texts_score_high() {
        let dims = 16;
        let toks = embed_tokens("rust retrieval engine", dims);
        let score = maxsim_score(&toks, &toks);
        assert!(score > 2.5, "expected high self-MaxSim, got {score}");
    }

    #[tokio::test]
    async fn ranks_lexical_overlap_higher() {
        let rr = HashMaxSimReranker::new(24);
        let cands = vec![
            scored("a", "baking sourdough bread recipes"),
            scored("b", "rust hybrid retrieval engine bm25"),
        ];
        let out = rr
            .rerank("rust retrieval", &cands, 2)
            .await
            .expect("rerank");
        assert_eq!(out[0].chunk.id.as_str(), "b");
    }

    fn scored(id: &str, text: &str) -> ScoredChunk {
        ScoredChunk {
            chunk: Chunk {
                id: ChunkId::new(id),
                document_id: DocumentId::new("d"),
                collection_id: CollectionId::new("c"),
                text: text.into(),
                ordinal: 0,
                metadata: HashMap::new(),
            },
            scores: ScoreBreakdown {
                final_score: 0.0,
                ..ScoreBreakdown::default()
            },
        }
    }
}
