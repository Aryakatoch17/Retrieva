//! CE vs ColBERT-ONNX MaxSim quality+latency on a shared candidate pool (BEIR).

use crate::beir::BeirDataset;
use crate::eval::BenchEmbedder;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How first-stage candidates are built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSource {
    /// BM25 + dense hybrid fuse.
    Hybrid,
    /// Dense ANN only.
    Dense,
}

/// One arm of the quality/latency table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankArmStats {
    /// Backend label (`identity` | `cross_encoder` | `maxsim_onnx`).
    pub backend: String,
    /// Candidate source label.
    pub candidate_source: String,
    /// Mean nDCG@k over evaluated queries.
    pub ndcg_at_k: f64,
    /// Mean rerank latency (ms) over queries (0 for identity).
    pub mean_latency_ms: f64,
    /// Queries scored.
    pub n_queries: usize,
}

/// Headline CE vs MaxSim report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankAblateReport {
    /// Dataset path.
    pub dataset: String,
    /// Candidate pool size (shared).
    pub candidate_pool: usize,
    /// Ranking cutoff for nDCG.
    pub top_k: usize,
    /// How candidates were built (primary).
    pub candidate_source: String,
    /// Arms.
    pub arms: Vec<RerankArmStats>,
    /// Notes.
    pub notes: Vec<String>,
}

/// Run dense / hybrid baselines + CE / MaxSim on BEIR.
#[allow(clippy::too_many_arguments)]
pub async fn run_ce_vs_maxsim_beir(
    data_dir: &Path,
    dataset: &BeirDataset,
    top_k: usize,
    candidate_pool: usize,
    embedder: BenchEmbedder,
    dims: usize,
    onnx_embed_model: Option<PathBuf>,
    ce_path: PathBuf,
    maxsim_path: PathBuf,
) -> Result<RerankAblateReport> {
    run_rerank_ablate(
        data_dir,
        dataset,
        top_k,
        candidate_pool,
        embedder,
        dims,
        onnx_embed_model,
        ce_path,
        maxsim_path,
        &[CandidateSource::Hybrid, CandidateSource::Dense],
        true,
    )
    .await
}

/// Full ablate with selectable candidate sources and optional MaxSim precompute.
#[allow(clippy::too_many_arguments)]
pub async fn run_rerank_ablate(
    data_dir: &Path,
    dataset: &BeirDataset,
    top_k: usize,
    candidate_pool: usize,
    embedder: BenchEmbedder,
    dims: usize,
    onnx_embed_model: Option<PathBuf>,
    ce_path: PathBuf,
    maxsim_path: PathBuf,
    sources: &[CandidateSource],
    precompute_maxsim_docs: bool,
) -> Result<RerankAblateReport> {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (
            data_dir,
            dataset,
            top_k,
            candidate_pool,
            embedder,
            dims,
            onnx_embed_model,
            ce_path,
            maxsim_path,
            sources,
            precompute_maxsim_docs,
        );
        anyhow::bail!("run_rerank_ablate requires `--features onnx`");
    }
    #[cfg(feature = "onnx")]
    {
        use crate::eval::{run_beir_eval, EvalOptions};
        use crate::metrics::mean_ndcg_at_k;
        use anyhow::Context;
        use retrieva_core::traits::{Embedder, Fusion, Reranker, SparseIndex};
        use retrieva_core::{CollectionId, Config, EmbeddingConfig, RerankConfig, SearchMode};
        use retrieva_fusion::fusion_from_config;
        use retrieva_index::{TantivySparseIndex, UsearchVectorIndex};
        use retrieva_ingest::{chunk_text, HashEmbedder, OnnxEmbedder};
        use retrieva_rerank::{CrossEncoderReranker, IdentityReranker, OnnxMaxSimReranker};
        use std::collections::HashMap;
        use std::sync::Arc;
        use std::time::Instant;
        use tracing::info;

        let eval_opts = EvalOptions {
            collection: "rerank_ablate".into(),
            mode: SearchMode::Bm25,
            top_k,
            candidate_pool,
            use_cache: false,
            dimensions: dims,
            skip_index: false,
            embedder,
            onnx_model: onnx_embed_model.clone(),
            onnx_batch_size: 32,
        };
        let _ = std::fs::remove_dir_all(data_dir);
        let _warmup = run_beir_eval(data_dir, dataset, &eval_opts)
            .await
            .context("index corpus for rerank ablate")?;

        let collection = CollectionId::new("rerank_ablate");
        let sparse = TantivySparseIndex::open(data_dir).context("open BM25")?;
        let vectors = UsearchVectorIndex::open(data_dir, dims).context("open ANN")?;

        let (emb, emb_note): (Arc<dyn Embedder>, String) = match embedder {
            BenchEmbedder::Hash => (
                Arc::new(HashEmbedder::new(dims)),
                "hash-dense (not a quality claim for retrieval)".into(),
            ),
            BenchEmbedder::Onnx => {
                let model = onnx_embed_model
                    .unwrap_or_else(|| PathBuf::from("models/bge-small-en-v1.5/model.onnx"));
                let cfg = EmbeddingConfig {
                    model_path: model,
                    model_id: "BAAI/bge-small-en-v1.5".into(),
                    dimensions: dims,
                    batch_size: 32,
                };
                (
                    Arc::new(OnnxEmbedder::from_config(&cfg)?.with_query_mode(true)),
                    "onnx/bge-small".into(),
                )
            }
        };

        let mut cfg = Config::default();
        cfg.search.fusion = "weighted".into();
        cfg.search.bm25_weight = 0.5;
        cfg.search.dense_weight = 0.5;
        let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&cfg.search)?);

        let mut ce_cfg = RerankConfig::default();
        ce_cfg.cross_encoder_path = ce_path;
        ce_cfg.backend = "cross_encoder".into();
        let ce: Arc<dyn Reranker> = Arc::new(CrossEncoderReranker::from_config(&ce_cfg)?);

        let mut ms_cfg = RerankConfig::default();
        ms_cfg.maxsim_path = maxsim_path;
        ms_cfg.backend = "maxsim".into();
        let maxsim = Arc::new(OnnxMaxSimReranker::from_config(&ms_cfg)?);

        if precompute_maxsim_docs {
            let mut pairs: Vec<(String, String)> = Vec::new();
            for (doc_id, doc) in &dataset.corpus {
                let text = doc.full_text();
                for (ord, part) in chunk_text(&text, 512, 64).into_iter().enumerate() {
                    pairs.push((format!("{doc_id}#{ord}"), part));
                }
            }
            let refs: Vec<(&str, &str)> = pairs
                .iter()
                .map(|(id, t)| (id.as_str(), t.as_str()))
                .collect();
            info!(chunks = refs.len(), "precomputing MaxSim doc tokens");
            maxsim.precompute_chunks(refs)?;
        }

        let maxsim_r: Arc<dyn Reranker> = maxsim;
        let identity: Arc<dyn Reranker> = Arc::new(IdentityReranker);

        let sources = if sources.is_empty() {
            vec![CandidateSource::Hybrid]
        } else {
            sources.to_vec()
        };

        let mut arm_defs: Vec<(String, String, Arc<dyn Reranker>)> = Vec::new();
        for src in &sources {
            let src_label = match src {
                CandidateSource::Hybrid => "hybrid",
                CandidateSource::Dense => "dense",
            };
            arm_defs.push((
                format!("{src_label}+identity"),
                src_label.into(),
                Arc::clone(&identity),
            ));
            arm_defs.push((
                format!("{src_label}+maxsim_onnx"),
                src_label.into(),
                Arc::clone(&maxsim_r),
            ));
            arm_defs.push((
                format!("{src_label}+cross_encoder"),
                src_label.into(),
                Arc::clone(&ce),
            ));
        }

        let mut arm_rankings: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        let mut arm_latency: HashMap<String, Vec<f64>> = HashMap::new();
        for (name, _, _) in &arm_defs {
            arm_rankings.insert(name.clone(), HashMap::new());
            arm_latency.insert(name.clone(), Vec::new());
        }

        let mut n_queries = 0usize;
        for (qid, q) in &dataset.queries {
            if !dataset.qrels.contains_key(qid) {
                continue;
            }
            n_queries += 1;
            let q_emb = emb.embed(&q.text).await.context("embed")?;
            let dense = vectors
                .search_chunks(&collection, &q_emb.values, candidate_pool)
                .await
                .context("dense")?;
            let bm25 = sparse
                .search(&collection, &q.text, candidate_pool)
                .await
                .context("bm25")?;
            let hybrid = fusion
                .fuse(&[bm25, dense.clone()], candidate_pool)
                .context("fuse")?;

            for (name, src_label, reranker) in &arm_defs {
                let candidates = if src_label == "dense" {
                    dense.clone()
                } else {
                    hybrid.clone()
                };
                let t0 = Instant::now();
                let ranked = if name.ends_with("+identity") {
                    let mut c = candidates;
                    c.truncate(top_k);
                    c
                } else {
                    reranker
                        .rerank(&q.text, &candidates, top_k)
                        .await
                        .with_context(|| format!("rerank {name}"))?
                };
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                arm_latency.get_mut(name).unwrap().push(ms);

                let mut seen = HashMap::new();
                let mut docs = Vec::new();
                for hit in ranked {
                    let did = hit.chunk.document_id.as_str().to_string();
                    if seen.insert(did.clone(), ()).is_none() {
                        docs.push(did);
                    }
                }
                arm_rankings
                    .get_mut(name)
                    .unwrap()
                    .insert(qid.clone(), docs);
            }
            if n_queries % 50 == 0 {
                info!(n_queries, "rerank ablate progress");
            }
        }

        let mut stats = Vec::new();
        for (name, src_label, _) in &arm_defs {
            let rankings = arm_rankings.get(name).unwrap();
            let lats = arm_latency.get(name).unwrap();
            let mean_lat = if lats.is_empty() {
                0.0
            } else {
                lats.iter().sum::<f64>() / lats.len() as f64
            };
            let backend = name
                .rsplit_once('+')
                .map(|(_, b)| b.to_string())
                .unwrap_or_else(|| name.clone());
            stats.push(RerankArmStats {
                backend,
                candidate_source: src_label.clone(),
                ndcg_at_k: mean_ndcg_at_k(&dataset.qrels, rankings, top_k),
                mean_latency_ms: mean_lat,
                n_queries,
            });
        }

        Ok(RerankAblateReport {
            dataset: dataset.root.display().to_string(),
            candidate_pool,
            top_k,
            candidate_source: format!("{sources:?} / {emb_note}"),
            arms: stats,
            notes: vec![
                "identity = first-stage only (no rerank).".into(),
                "maxsim_onnx = answerai-colbert-small-v1 Vespa ONNX with doc-token cache.".into(),
                "cross_encoder = bge-reranker-base.".into(),
                "Latency is mean per-query rerank over the candidate pool (CPU unless --features cuda).".into(),
                "Hash MaxSim is not included (smoke-only; near-random nDCG).".into(),
            ],
        })
    }
}

/// Write JSON + short markdown table.
pub fn write_rerank_ablate(out_dir: &Path, report: &RerankAblateReport) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join("ablate_ce_vs_maxsim_beir.json");
    std::fs::write(&path, serde_json::to_string_pretty(report)?)?;
    let mut md = String::from("# CE vs ColBERT MaxSim (BEIR)\n\n");
    md.push_str(&format!(
        "dataset={} pool={} top_k={} candidates={}\n\n",
        report.dataset, report.candidate_pool, report.top_k, report.candidate_source
    ));
    md.push_str("| Candidates | Backend | nDCG@");
    md.push_str(&report.top_k.to_string());
    md.push_str(
        " | mean latency (ms) | n |\n|-----------|---------|--------|------------------|---|\n",
    );
    for a in &report.arms {
        md.push_str(&format!(
            "| {} | {} | {:.4} | {:.2} | {} |\n",
            a.candidate_source, a.backend, a.ndcg_at_k, a.mean_latency_ms, a.n_queries
        ));
    }
    for n in &report.notes {
        md.push_str(&format!("\n- {n}\n"));
    }
    std::fs::write(out_dir.join("ablate_ce_vs_maxsim_beir.md"), md)?;
    Ok(path)
}
