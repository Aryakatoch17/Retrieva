//! Ablation helper: run the same candidates through multiple rerankers.

use retrieva_core::traits::Reranker;
use retrieva_core::{Result, ScoredChunk};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

/// One backend's ranked id list + latency.
#[derive(Debug, Clone, Serialize)]
pub struct AblationArm {
    /// Backend name (`identity`, `maxsim`, …).
    pub backend: String,
    /// Ranked chunk ids (best first).
    pub ranking: Vec<String>,
    /// Top score after rerank.
    pub top_score: Option<f32>,
    /// Wall time in milliseconds.
    pub latency_ms: f64,
}

/// Full ablation report for one query.
#[derive(Debug, Clone, Serialize)]
pub struct AblationReport {
    /// Query text.
    pub query: String,
    /// Number of candidates fed to each arm.
    pub candidate_count: usize,
    /// Per-backend results.
    pub arms: Vec<AblationArm>,
}

/// Run each `(name, reranker)` on the same `candidates`, keeping `top_k`.
pub async fn ablate_rerankers(
    query: &str,
    candidates: &[ScoredChunk],
    top_k: usize,
    arms: &[(String, Arc<dyn Reranker>)],
) -> Result<AblationReport> {
    let mut out = Vec::with_capacity(arms.len());
    for (name, reranker) in arms {
        let t0 = Instant::now();
        let ranked = reranker.rerank(query, candidates, top_k).await?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        out.push(AblationArm {
            backend: name.clone(),
            ranking: ranked
                .iter()
                .map(|h| h.chunk.id.as_str().to_string())
                .collect(),
            top_score: ranked.first().map(|h| h.scores.final_score),
            latency_ms,
        });
    }
    Ok(AblationReport {
        query: query.to_string(),
        candidate_count: candidates.len(),
        arms: out,
    })
}

/// Pairwise top-1 agreement rate across arms (for logging / tests).
pub fn top1_agreement(report: &AblationReport) -> f32 {
    if report.arms.len() < 2 {
        return 1.0;
    }
    let first = report.arms[0].ranking.first();
    let mut agree = 0usize;
    let mut total = 0usize;
    for arm in report.arms.iter().skip(1) {
        total += 1;
        if arm.ranking.first() == first {
            agree += 1;
        }
    }
    if total == 0 {
        1.0
    } else {
        agree as f32 / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::IdentityReranker;
    use crate::maxsim::HashMaxSimReranker;
    use retrieva_core::{Chunk, ChunkId, CollectionId, DocumentId, ScoreBreakdown};
    use std::collections::HashMap;

    #[tokio::test]
    async fn ablation_runs_two_arms() {
        let cands = vec![
            scored("a", "zzzz unrelated filler text", 0.1),
            scored("b", "rust retrieval hybrid engine rust retrieval", 0.9),
        ];
        let arms: Vec<(String, Arc<dyn Reranker>)> = vec![
            ("identity".into(), Arc::new(IdentityReranker)),
            ("maxsim".into(), Arc::new(HashMaxSimReranker::new(32))),
        ];
        let report = ablate_rerankers("rust retrieval", &cands, 2, &arms)
            .await
            .expect("ablate");
        assert_eq!(report.arms.len(), 2);
        assert_eq!(report.arms[0].ranking[0], "b"); // identity keeps final_score order
        let maxsim = &report.arms[1];
        assert_eq!(
            maxsim.ranking[0], "b",
            "maxsim ranking={:?} scores_top={:?}",
            maxsim.ranking, maxsim.top_score
        );
    }

    fn scored(id: &str, text: &str, score: f32) -> ScoredChunk {
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
                final_score: score,
                ..ScoreBreakdown::default()
            },
        }
    }
}
