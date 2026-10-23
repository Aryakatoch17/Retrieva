//! Rank fusion strategies: Reciprocal Rank Fusion (RRF) and weighted score fusion.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use retrieva_core::traits::Fusion;
use retrieva_core::{Result, RetrievaError, ScoredChunk, SearchConfig};
use std::collections::HashMap;

/// Reciprocal Rank Fusion: `score += 1 / (k + rank + 1)` across lists.
#[derive(Debug, Clone)]
pub struct RrfFusion {
    /// RRF constant (standard default 60).
    pub k: u32,
}

impl Default for RrfFusion {
    fn default() -> Self {
        Self { k: 60 }
    }
}

impl RrfFusion {
    /// Create an RRF fusion with the given `k`.
    pub fn new(k: u32) -> Self {
        Self { k }
    }
}

impl Fusion for RrfFusion {
    fn fuse(&self, lists: &[Vec<ScoredChunk>], top_k: usize) -> Result<Vec<ScoredChunk>> {
        let mut scores: HashMap<String, (ScoredChunk, f32)> = HashMap::new();
        let k = self.k as f32;
        for list in lists {
            for (rank, hit) in list.iter().enumerate() {
                let contribution = 1.0 / (k + rank as f32 + 1.0);
                let entry = scores
                    .entry(hit.chunk.id.as_str().to_string())
                    .or_insert_with(|| (hit.clone(), 0.0));
                entry.1 += contribution;
                merge_stage_scores(&mut entry.0, hit);
            }
        }
        Ok(finalize(scores, top_k))
    }
}

/// Weighted linear combination of per-list `final_score` values.
///
/// List `i` is scaled by `weights[i]` (missing weights default to `1.0`).
/// Stage scores (`bm25`, `dense`, …) are preserved from contributing hits.
#[derive(Debug, Clone)]
pub struct WeightedFusion {
    /// Per-list weights (aligned with `lists` order).
    pub weights: Vec<f32>,
}

impl WeightedFusion {
    /// Create a weighted fusion.
    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    /// Convenience for the common BM25 + dense pair.
    pub fn bm25_dense(bm25_weight: f32, dense_weight: f32) -> Self {
        Self {
            weights: vec![bm25_weight, dense_weight],
        }
    }
}

impl Fusion for WeightedFusion {
    fn fuse(&self, lists: &[Vec<ScoredChunk>], top_k: usize) -> Result<Vec<ScoredChunk>> {
        let mut scores: HashMap<String, (ScoredChunk, f32)> = HashMap::new();
        for (list_idx, list) in lists.iter().enumerate() {
            let w = self.weights.get(list_idx).copied().unwrap_or(1.0);
            for hit in list {
                let contribution = w * hit.scores.final_score;
                let entry = scores
                    .entry(hit.chunk.id.as_str().to_string())
                    .or_insert_with(|| (hit.clone(), 0.0));
                entry.1 += contribution;
                merge_stage_scores(&mut entry.0, hit);
            }
        }
        Ok(finalize(scores, top_k))
    }
}

/// Build a [`Fusion`] implementation from search config (`rrf` or `weighted`).
pub fn fusion_from_config(cfg: &SearchConfig) -> Result<Box<dyn Fusion>> {
    match cfg.fusion.as_str() {
        "rrf" => Ok(Box::new(RrfFusion::new(cfg.rrf_k))),
        "weighted" => Ok(Box::new(WeightedFusion::bm25_dense(
            cfg.bm25_weight,
            cfg.dense_weight,
        ))),
        other => Err(RetrievaError::InvalidArgument(format!(
            "unknown fusion strategy '{other}' (expected rrf|weighted)"
        ))),
    }
}

fn merge_stage_scores(dst: &mut ScoredChunk, src: &ScoredChunk) {
    if let Some(bm25) = src.scores.bm25 {
        dst.scores.bm25 = Some(bm25);
    }
    if let Some(dense) = src.scores.dense {
        dst.scores.dense = Some(dense);
    }
    if let Some(rerank) = src.scores.rerank {
        dst.scores.rerank = Some(rerank);
    }
}

fn finalize(scores: HashMap<String, (ScoredChunk, f32)>, top_k: usize) -> Vec<ScoredChunk> {
    let mut fused: Vec<ScoredChunk> = scores
        .into_values()
        .map(|(mut hit, fused)| {
            hit.scores.fused = Some(fused);
            hit.scores.final_score = fused;
            hit
        })
        .collect();
    fused.sort_by(|a, b| {
        b.scores
            .final_score
            .partial_cmp(&a.scores.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(top_k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{Chunk, ChunkId, CollectionId, DocumentId, ScoreBreakdown};
    use std::collections::HashMap as StdHashMap;

    fn hit(id: &str, score: f32, bm25: Option<f32>, dense: Option<f32>) -> ScoredChunk {
        ScoredChunk {
            chunk: Chunk {
                id: ChunkId::new(id),
                document_id: DocumentId::new("d"),
                collection_id: CollectionId::new("c"),
                text: id.into(),
                ordinal: 0,
                metadata: StdHashMap::new(),
            },
            scores: ScoreBreakdown {
                bm25,
                dense,
                final_score: score,
                ..ScoreBreakdown::default()
            },
        }
    }

    #[test]
    fn rrf_prefers_multi_list_agreement() {
        let fusion = RrfFusion::new(60);
        // y is rank-0 in both lists → highest RRF.
        let a = vec![
            hit("y", 10.0, Some(10.0), None),
            hit("z", 5.0, Some(5.0), None),
            hit("x", 1.0, Some(1.0), None),
        ];
        let b = vec![
            hit("y", 9.0, None, Some(9.0)),
            hit("x", 3.0, None, Some(3.0)),
            hit("z", 1.0, None, Some(1.0)),
        ];
        let out = fusion.fuse(&[a, b], 3).expect("fuse");
        assert_eq!(out[0].chunk.id.as_str(), "y");
        assert!(out[0].scores.fused.unwrap() > out[1].scores.fused.unwrap());
        assert!(out[0].scores.bm25.is_some());
        assert!(out[0].scores.dense.is_some());
    }

    #[test]
    fn weighted_scales_scores() {
        let fusion = WeightedFusion::bm25_dense(1.0, 0.0);
        let a = vec![hit("bm25-win", 100.0, Some(100.0), None)];
        let b = vec![hit("dense-win", 100.0, None, Some(100.0))];
        let out = fusion.fuse(&[a, b], 2).expect("fuse");
        assert_eq!(out[0].chunk.id.as_str(), "bm25-win");
    }
}
