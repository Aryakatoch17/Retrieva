//! Multi-mode BEIR matrix + honest hybrid gate.

use crate::beir::BeirDataset;
use crate::eval::{run_beir_eval, BenchEmbedder, EvalOptions};
use crate::report::write_eval;
use anyhow::{bail, Context, Result};
use retrieva_core::SearchMode;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One row in a mode comparison table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeRow {
    /// Dataset name.
    pub dataset: String,
    /// Mode label.
    pub mode: String,
    /// nDCG@k.
    pub ndcg_at_k: f64,
    /// Recall@k.
    pub recall_at_k: f64,
    /// MRR.
    pub mrr: f64,
}

/// Full matrix report across datasets × modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixReport {
    /// Flat rows.
    pub rows: Vec<ModeRow>,
    /// True only when hybrid shows a measurable gain over both singles (see notes).
    pub hybrid_ok: bool,
    /// Human notes (per-dataset verdict).
    pub notes: Vec<String>,
    /// Embedder used (`hash` | `onnx`).
    #[serde(default)]
    pub embedder: String,
}

/// Modes used in the standard matrix.
///
/// `HybridRerank` is omitted: hash MaxSim is a smoke test, not a quality row.
/// Real CE / ColBERT numbers come from `rerank-ablate`.
pub fn default_modes() -> Vec<SearchMode> {
    vec![SearchMode::Bm25, SearchMode::Dense, SearchMode::Hybrid]
}

/// Options for a matrix run.
#[derive(Debug, Clone)]
pub struct MatrixOptions {
    /// Ranking cutoff.
    pub top_k: usize,
    /// Candidate pool.
    pub candidate_pool: usize,
    /// Embedder backend.
    pub embedder: BenchEmbedder,
    /// Embedding dims (32 hash / 384 BGE).
    pub dimensions: usize,
    /// ONNX model path.
    pub onnx_model: Option<PathBuf>,
    /// ONNX batch size.
    pub onnx_batch_size: usize,
    /// Minimum nDCG gain over max(bm25, dense) to call hybrid a win.
    pub min_hybrid_gain: f64,
}

impl Default for MatrixOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            candidate_pool: 50,
            embedder: BenchEmbedder::Hash,
            dimensions: 32,
            onnx_model: None,
            onnx_batch_size: 32,
            min_hybrid_gain: 0.01,
        }
    }
}

/// Run all modes on one dataset (reuses one indexed data_dir).
pub async fn run_dataset_matrix(
    data_dir: &Path,
    dataset: &BeirDataset,
    out_dir: &Path,
    name: &str,
    opts: &MatrixOptions,
) -> Result<Vec<ModeRow>> {
    let mut rows = Vec::new();
    let _ = std::fs::remove_dir_all(data_dir);
    let modes = default_modes();
    for (i, mode) in modes.into_iter().enumerate() {
        if i > 0 && !data_dir.exists() {
            bail!("index disappeared mid-matrix");
        }
        let eval = EvalOptions {
            collection: name.to_string(),
            mode,
            top_k: opts.top_k,
            candidate_pool: opts.candidate_pool,
            use_cache: false,
            dimensions: opts.dimensions,
            skip_index: i > 0,
            embedder: opts.embedder,
            onnx_model: opts.onnx_model.clone(),
            onnx_batch_size: opts.onnx_batch_size,
        };
        let report = run_beir_eval(data_dir, dataset, &eval)
            .await
            .with_context(|| format!("eval {name} {:?}", eval.mode))?;
        let mode_label = format!("{:?}", eval.mode);
        let row = ModeRow {
            dataset: name.to_string(),
            mode: mode_label.clone(),
            ndcg_at_k: report.ndcg_at_k,
            recall_at_k: report.recall_at_k,
            mrr: report.mrr,
        };
        let mut stamped = report;
        stamped.suite = "matrix".into();
        let safe = mode_label.replace('+', "_");
        write_eval(out_dir, &format!("{name}_{safe}"), &stamped)?;
        rows.push(row);
    }
    Ok(rows)
}

/// Honest hybrid verdict per dataset.
///
/// - **hash** embedder: never `hybrid_ok`; note that the run cannot speak to hybrid quality.
/// - **onnx**: `win` if hybrid ≥ max(bm25, dense) + `min_gain`; `mixed` if within tolerance of
///   the best single but no clear gain (common on SciFact/NFCorpus where BM25 is strong);
///   `fail` if hybrid is worse than max − eps.
pub fn assess_hybrid(
    rows: &[ModeRow],
    embedder: BenchEmbedder,
    min_gain: f64,
    eps: f64,
) -> (bool, Vec<String>) {
    let mut notes = Vec::new();
    let mut all_win = true;
    let mut datasets: Vec<String> = rows.iter().map(|r| r.dataset.clone()).collect();
    datasets.sort();
    datasets.dedup();

    if embedder == BenchEmbedder::Hash {
        for ds in &datasets {
            let (b, d, h) = extract_trio(rows, ds);
            notes.push(format!(
                "{ds}: hybrid={h:.4} bm25={b:.4} dense={d:.4} — HASH EMBEDDER; \
                 hybrid≈BM25 is expected and not a hybrid finding (inconclusive)"
            ));
        }
        notes.insert(
            0,
            "hybrid_ok=false: hash-dense carries no semantic signal; re-run with --embedder onnx"
                .into(),
        );
        return (false, notes);
    }

    for ds in datasets {
        let (b, d, h) = extract_trio(rows, &ds);
        let best = b.max(d);
        if h + 1e-12 < best - eps {
            all_win = false;
            notes.push(format!(
                "{ds}: FAIL hybrid={h:.4} < max(bm25={b:.4}, dense={d:.4}) - {eps}"
            ));
        } else if h + 1e-12 >= best + min_gain {
            notes.push(format!(
                "{ds}: WIN hybrid={h:.4} beats max(bm25={b:.4}, dense={d:.4}) by ≥ {min_gain}"
            ));
        } else {
            all_win = false;
            notes.push(format!(
                "{ds}: MIXED hybrid={h:.4} bm25={b:.4} dense={d:.4} \
                 (no ≥{min_gain} gain over best single; normal when BM25 is strong)"
            ));
        }
    }
    (all_win, notes)
}

fn extract_trio(rows: &[ModeRow], ds: &str) -> (f64, f64, f64) {
    let mut bm25 = 0.0;
    let mut dense = 0.0;
    let mut hybrid = 0.0;
    for r in rows.iter().filter(|r| r.dataset == ds) {
        match r.mode.as_str() {
            "Bm25" => bm25 = r.ndcg_at_k,
            "Dense" => dense = r.ndcg_at_k,
            "Hybrid" => hybrid = r.ndcg_at_k,
            _ => {}
        }
    }
    (bm25, dense, hybrid)
}

/// Deprecated soft gate kept for API compat; prefer [`assess_hybrid`].
pub fn assert_hybrid_beats_singles(rows: &[ModeRow], eps: f64) -> Result<Vec<String>> {
    let (_ok, notes) = assess_hybrid(rows, BenchEmbedder::Onnx, 0.01, eps);
    // Do not bail on mixed — callers set hybrid_ok from assess_hybrid.
    Ok(notes)
}

/// Write markdown table for README.
pub fn matrix_markdown(rows: &[ModeRow], top_k: usize) -> String {
    let mut out = String::from("| Dataset | Mode | nDCG@");
    out.push_str(&top_k.to_string());
    out.push_str(" | Recall@");
    out.push_str(&top_k.to_string());
    out.push_str(" | MRR |\n|---------|------|--------|---------|-----|\n");
    for r in rows {
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} |\n",
            r.dataset, r.mode, r.ndcg_at_k, r.recall_at_k, r.mrr
        ));
    }
    out
}

/// Persist matrix JSON + MD.
pub fn write_matrix(out_dir: &Path, report: &MatrixReport) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join("matrix.json");
    std::fs::write(&path, serde_json::to_string_pretty(report)?)?;
    let md = out_dir.join("matrix.md");
    let table = matrix_markdown(&report.rows, 10);
    let notes = report.notes.join("\n- ");
    let emb = &report.embedder;
    std::fs::write(
        &md,
        format!(
            "# BEIR mode matrix ({emb})\n\nhybrid_ok={} (measurable gain over both singles)\n\n{table}\n\n## Hybrid assessment\n\n- {notes}\n",
            report.hybrid_ok
        ),
    )?;
    Ok(path)
}
