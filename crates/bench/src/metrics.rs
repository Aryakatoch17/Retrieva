//! Ranking metrics: nDCG@k, Recall@k, MRR.

use std::collections::HashMap;

/// Graded relevance: doc_id → relevance (≥ 1 is relevant for binary metrics).
pub type Qrels = HashMap<String, HashMap<String, u32>>;

/// Compute mean nDCG@k over queries.
pub fn mean_ndcg_at_k(qrels: &Qrels, rankings: &HashMap<String, Vec<String>>, k: usize) -> f64 {
    mean_over_queries(qrels, rankings, |rels, ranked| ndcg_at_k(rels, ranked, k))
}

/// Compute mean Recall@k (binary: relevance > 0) over queries.
pub fn mean_recall_at_k(qrels: &Qrels, rankings: &HashMap<String, Vec<String>>, k: usize) -> f64 {
    mean_over_queries(qrels, rankings, |rels, ranked| recall_at_k(rels, ranked, k))
}

/// Mean reciprocal rank of the first relevant document.
pub fn mean_mrr(qrels: &Qrels, rankings: &HashMap<String, Vec<String>>) -> f64 {
    mean_over_queries(qrels, rankings, mrr)
}

fn mean_over_queries(
    qrels: &Qrels,
    rankings: &HashMap<String, Vec<String>>,
    f: impl Fn(&HashMap<String, u32>, &[String]) -> f64,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (qid, rels) in qrels {
        if rels.is_empty() {
            continue;
        }
        let ranked = rankings.get(qid).map(Vec::as_slice).unwrap_or(&[]);
        sum += f(rels, ranked);
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// nDCG@k for a single query.
pub fn ndcg_at_k(rels: &HashMap<String, u32>, ranked: &[String], k: usize) -> f64 {
    let k = k.max(1);
    let dcg = dcg_at_k(rels, ranked, k);
    let mut ideal: Vec<u32> = rels.values().copied().collect();
    ideal.sort_by(|a, b| b.cmp(a));
    let idcg = dcg_from_gains(&ideal, k);
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn dcg_at_k(rels: &HashMap<String, u32>, ranked: &[String], k: usize) -> f64 {
    let gains: Vec<u32> = ranked
        .iter()
        .take(k)
        .map(|id| rels.get(id).copied().unwrap_or(0))
        .collect();
    dcg_from_gains(&gains, k)
}

fn dcg_from_gains(gains: &[u32], k: usize) -> f64 {
    gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &g)| {
            let gain = g as f64;
            gain / ((i as f64) + 2.0).log2()
        })
        .sum()
}

/// Recall@k: |relevant ∩ top-k| / |relevant|.
pub fn recall_at_k(rels: &HashMap<String, u32>, ranked: &[String], k: usize) -> f64 {
    let relevant: Vec<&String> = rels
        .iter()
        .filter(|(_, &r)| r > 0)
        .map(|(id, _)| id)
        .collect();
    if relevant.is_empty() {
        return 0.0;
    }
    let hit = ranked
        .iter()
        .take(k)
        .filter(|id| rels.get(*id).copied().unwrap_or(0) > 0)
        .count();
    hit as f64 / relevant.len() as f64
}

/// Reciprocal rank of first relevant hit (0 if none).
pub fn mrr(rels: &HashMap<String, u32>, ranked: &[String]) -> f64 {
    for (i, id) in ranked.iter().enumerate() {
        if rels.get(id).copied().unwrap_or(0) > 0 {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking_ndcg_one() {
        let mut rels = HashMap::new();
        rels.insert("a".into(), 2);
        rels.insert("b".into(), 1);
        let ranked = vec!["a".into(), "b".into(), "c".into()];
        assert!((ndcg_at_k(&rels, &ranked, 2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recall_partial() {
        let mut rels = HashMap::new();
        rels.insert("a".into(), 1);
        rels.insert("b".into(), 1);
        let ranked = vec!["x".into(), "a".into()];
        assert!((recall_at_k(&rels, &ranked, 2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mrr_second_hit() {
        let mut rels = HashMap::new();
        rels.insert("b".into(), 1);
        let ranked = vec!["a".into(), "b".into()];
        assert!((mrr(&rels, &ranked) - 0.5).abs() < 1e-9);
    }
}
