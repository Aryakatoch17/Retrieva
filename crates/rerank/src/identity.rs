//! Pass-through reranker that sorts by existing `final_score`.

use async_trait::async_trait;
use retrieva_core::traits::Reranker;
use retrieva_core::{Result, ScoredChunk};

/// No-model reranker: keep ranking order, optionally stamp a synthetic `rerank` score.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityReranker;

#[async_trait]
impl Reranker for IdentityReranker {
    async fn rerank(
        &self,
        _query: &str,
        candidates: &[ScoredChunk],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        let mut out = candidates.to_vec();
        out.sort_by(|a, b| {
            b.scores
                .final_score
                .partial_cmp(&a.scores.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(top_k);
        for (i, hit) in out.iter_mut().enumerate() {
            hit.scores.rerank = Some(1.0 / (i as f32 + 1.0));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::{Chunk, ChunkId, CollectionId, DocumentId, ScoreBreakdown};
    use std::collections::HashMap;

    #[tokio::test]
    async fn truncates_and_orders() {
        let cands = vec![scored("a", 0.2), scored("b", 0.9), scored("c", 0.5)];
        let out = IdentityReranker
            .rerank("q", &cands, 2)
            .await
            .expect("rerank");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk.id.as_str(), "b");
        assert!(out[0].scores.rerank.is_some());
    }

    fn scored(id: &str, score: f32) -> ScoredChunk {
        ScoredChunk {
            chunk: Chunk {
                id: ChunkId::new(id),
                document_id: DocumentId::new("d"),
                collection_id: CollectionId::new("c"),
                text: id.into(),
                ordinal: 0,
                metadata: HashMap::new(),
            },
            scores: ScoreBreakdown {
                final_score: score,
                ..ScoreBreakdown::default()
            },
        }
    }
}
