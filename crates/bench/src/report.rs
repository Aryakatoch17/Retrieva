//! JSON + Markdown result writers under `bench/results/`.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// One evaluation run summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Suite name (`smoke`, `beir`, `load`).
    pub suite: String,
    /// Dataset / fixture path.
    pub dataset: String,
    /// Search mode label.
    pub mode: String,
    /// Cutoff k for ranking metrics.
    pub top_k: usize,
    /// Number of evaluated queries.
    pub n_queries: usize,
    /// Mean nDCG@k.
    pub ndcg_at_k: f64,
    /// Mean Recall@k.
    pub recall_at_k: f64,
    /// Mean reciprocal rank.
    pub mrr: f64,
    /// Wall-clock search time total (ms).
    pub search_ms_total: f64,
    /// ISO timestamp.
    pub created_at: String,
}

/// Load-test latency summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadReport {
    /// Suite label.
    pub suite: String,
    /// Total completed requests.
    pub requests: usize,
    /// Concurrency level.
    pub concurrency: usize,
    /// Successful responses.
    pub ok: usize,
    /// Failed responses.
    pub err: usize,
    /// Latency percentiles (ms).
    pub p50_ms: f64,
    /// p95 latency ms.
    pub p95_ms: f64,
    /// p99 latency ms.
    pub p99_ms: f64,
    /// Mean latency ms.
    pub mean_ms: f64,
    /// Throughput (req/s).
    pub qps: f64,
    /// Whether the query cache was consulted / filled.
    #[serde(default)]
    pub use_cache: bool,
    /// Cache hits observed on responses (`cache_hit=true`).
    #[serde(default)]
    pub cache_hits: u64,
    /// Cache misses (successful searches that were not cache hits).
    #[serde(default)]
    pub cache_misses: u64,
    /// `cache_hits / (cache_hits + cache_misses)`.
    #[serde(default)]
    pub cache_hit_rate: f64,
    /// Query mix used (`round_robin` | `zipf`).
    #[serde(default)]
    pub query_mix: String,
    /// Dataset / hardware footnote (not a marketing headline).
    #[serde(default)]
    pub notes: String,
    /// ISO timestamp.
    pub created_at: String,
}

/// Comparison of two eval reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    /// Baseline path.
    pub baseline: String,
    /// Current path.
    pub current: String,
    /// Absolute deltas (current − baseline).
    pub delta_ndcg: f64,
    /// Recall delta.
    pub delta_recall: f64,
    /// MRR delta.
    pub delta_mrr: f64,
    /// Whether current is not worse on nDCG (within epsilon).
    pub ok: bool,
}

impl EvalReport {
    /// Stamp `created_at` with UTC now.
    pub fn stamp(mut self) -> Self {
        self.created_at = Utc::now().to_rfc3339();
        self
    }
}

impl LoadReport {
    /// Stamp `created_at`.
    pub fn stamp(mut self) -> Self {
        self.created_at = Utc::now().to_rfc3339();
        self
    }
}

/// Ensure `out_dir` exists and write `name.json` + `name.md`.
pub fn write_eval(out_dir: &Path, name: &str, report: &EvalReport) -> Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(out_dir)?;
    let json_path = out_dir.join(format!("{name}.json"));
    let md_path = out_dir.join(format!("{name}.md"));
    let json = serde_json::to_string_pretty(report)?;
    fs::write(&json_path, json).with_context(|| format!("write {}", json_path.display()))?;
    let md = format!(
        "# Eval: {name}\n\n\
         | field | value |\n|-------|-------|\n\
         | suite | {} |\n\
         | dataset | {} |\n\
         | mode | {} |\n\
         | top_k | {} |\n\
         | n_queries | {} |\n\
         | nDCG@{k} | {:.4} |\n\
         | Recall@{k} | {:.4} |\n\
         | MRR | {:.4} |\n\
         | search_ms_total | {:.1} |\n\
         | created_at | {} |\n",
        report.suite,
        report.dataset,
        report.mode,
        report.top_k,
        report.n_queries,
        report.ndcg_at_k,
        report.recall_at_k,
        report.mrr,
        report.search_ms_total,
        report.created_at,
        k = report.top_k,
    );
    fs::write(&md_path, md)?;
    Ok((json_path, md_path))
}

/// Write load-test report.
pub fn write_load(out_dir: &Path, name: &str, report: &LoadReport) -> Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(out_dir)?;
    let json_path = out_dir.join(format!("{name}.json"));
    let md_path = out_dir.join(format!("{name}.md"));
    fs::write(&json_path, serde_json::to_string_pretty(report)?)?;
    let md = format!(
        "# Load: {name}\n\n\
         requests={} concurrency={} cache={} hit_rate={:.1}% ({}/{}) mix={}\n\n\
         ok={} err={} p50={:.2}ms p95={:.2}ms p99={:.2}ms mean={:.2}ms qps={:.2}\n\n\
         notes: {}\n",
        report.requests,
        report.concurrency,
        report.use_cache,
        report.cache_hit_rate * 100.0,
        report.cache_hits,
        report.cache_hits + report.cache_misses,
        report.query_mix,
        report.ok,
        report.err,
        report.p50_ms,
        report.p95_ms,
        report.p99_ms,
        report.mean_ms,
        report.qps,
        report.notes,
    );
    fs::write(&md_path, md)?;
    Ok((json_path, md_path))
}

/// Compare two eval JSON reports; fail if nDCG regresses by more than `eps`.
pub fn compare_eval(baseline: &Path, current: &Path, eps: f64) -> Result<CompareReport> {
    let base: EvalReport = serde_json::from_str(&fs::read_to_string(baseline)?)?;
    let cur: EvalReport = serde_json::from_str(&fs::read_to_string(current)?)?;
    let delta_ndcg = cur.ndcg_at_k - base.ndcg_at_k;
    let delta_recall = cur.recall_at_k - base.recall_at_k;
    let delta_mrr = cur.mrr - base.mrr;
    Ok(CompareReport {
        baseline: baseline.display().to_string(),
        current: current.display().to_string(),
        delta_ndcg,
        delta_recall,
        delta_mrr,
        ok: delta_ndcg >= -eps,
    })
}

/// Percentile of a sorted ascending slice (linear interpolation not needed for smoke).
pub fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_ms.len() as f64 - 1.0)).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}
