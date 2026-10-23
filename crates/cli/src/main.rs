//! Retrieva CLI — Phase 9: BEIR bench, cache, WAL, hybrid query, REST+gRPC serve.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use retrieva_bench::{
    assess_hybrid, compare_eval, parse_mode, run_beir_eval, run_beir_load_pair,
    run_ce_vs_maxsim_beir, run_dataset_matrix, run_load_test, write_eval, write_load, write_matrix,
    write_rerank_ablate, BeirDataset, BenchEmbedder, EvalOptions, LoadOptions, MatrixOptions,
    MatrixReport,
};
use retrieva_cache::cache_from_config;
use retrieva_core::traits::{Embedder, Fusion, Reranker, Retriever, SparseIndex};
use retrieva_core::{
    init_logging, CollectionId, Config, DocumentId, LoggingOptions, SearchFilters, SearchMode,
    SearchRequest,
};
use retrieva_fusion::fusion_from_config;
use retrieva_index::{IncrementalIndex, TantivySparseIndex, UsearchVectorIndex};
use retrieva_ingest::{ingest_path, HashEmbedder};
use retrieva_rerank::{
    ablate_rerankers, reranker_from_config, HashMaxSimReranker, IdentityReranker,
};
use retrieva_server::{serve, HybridEngine};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Retrieva — self-hostable hybrid RAG retrieval engine.
#[derive(Debug, Parser)]
#[command(name = "retrieva", version, about, long_about = None)]
struct Cli {
    /// Path to a TOML config file (defaults to config/retrieva.toml when present).
    #[arg(short, long, global = true, env = "RETRIEVA_CONFIG")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QueryModeArg {
    Bm25,
    Dense,
    Hybrid,
    #[value(name = "hybrid+rerank")]
    HybridRerank,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EmbedderArg {
    Hash,
    Onnx,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RerankerArg {
    /// Use `rerank.backend` from config.
    Config,
    Identity,
    Maxsim,
    #[value(name = "cross_encoder")]
    CrossEncoder,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print resolved configuration as JSON.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Ingest documents (parse → chunk → WAL → BM25 + dense ANN).
    Ingest {
        #[arg(long)]
        collection: String,
        path: PathBuf,
        #[arg(long, value_enum, default_value_t = EmbedderArg::Hash)]
        embedder: EmbedderArg,
    },
    /// Delete a document (and its chunks) from store + indexes.
    Delete {
        #[arg(long)]
        collection: String,
        /// Document id to remove.
        #[arg(long)]
        document: String,
    },
    /// Replay uncheckpointed WAL entries (also runs automatically on open).
    Recover,
    /// Start the REST server.
    Serve,
    /// Run a one-shot query.
    Query {
        #[arg(long)]
        collection: String,
        query: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, value_enum, default_value_t = QueryModeArg::Hybrid)]
        mode: QueryModeArg,
        #[arg(long, value_enum, default_value_t = EmbedderArg::Hash)]
        embedder: EmbedderArg,
        /// Override reranker for `hybrid+rerank` (default: config).
        #[arg(long, value_enum, default_value_t = RerankerArg::Config)]
        reranker: RerankerArg,
    },
    /// Ablate identity vs MaxSim (and cross-encoder when onnx + model).
    Ablate {
        #[arg(long)]
        collection: String,
        query: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 50)]
        candidate_pool: usize,
    },
    /// Compare Rust ONNX embeddings to Python `embed_parity.py` JSON (requires `--features onnx`).
    EmbedParity {
        /// Repeat for a batch (padding / length coverage). Defaults to one short string if omitted.
        #[arg(long, action = clap::ArgAction::Append)]
        text: Vec<String>,
        /// JSON produced by `python scripts/embed_parity.py ...` (optional self-check only).
        #[arg(long)]
        python_json: Option<PathBuf>,
        /// Apply BGE query prefix (parity texts should use the same flag in Python).
        #[arg(long, default_value_t = false)]
        query: bool,
    },
    /// Run benchmarks (smoke | beir | load | compare | matrix | cache | rerank-ablate).
    Bench {
        /// Suite: `smoke`, `beir`, `load`, `compare`, `matrix`, `cache`, or `rerank-ablate`.
        #[arg(long, default_value = "smoke")]
        suite: String,
        /// BEIR dataset directory (`corpus.jsonl`, `queries.jsonl`, `qrels/`).
        /// Repeat for matrix multi-dataset runs, or set `RETRIEVA_BENCH_DATASETS`.
        #[arg(long, action = clap::ArgAction::Append)]
        dataset: Vec<PathBuf>,
        /// Search mode for eval (`bm25`, `dense`, `hybrid`, `hybrid+rerank`).
        #[arg(long, default_value = "bm25")]
        mode: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Load-test request count.
        #[arg(long, default_value_t = 50)]
        requests: usize,
        /// Load-test concurrency.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Output directory for JSON/Markdown reports.
        #[arg(long, default_value = "bench/results")]
        out_dir: PathBuf,
        /// Baseline eval JSON (`compare` suite).
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Current eval JSON (`compare` suite).
        #[arg(long)]
        current: Option<PathBuf>,
        /// Embedder for matrix / beir / load / rerank-ablate (`hash` or `onnx`).
        #[arg(long, value_enum, default_value_t = EmbedderArg::Hash)]
        embedder: EmbedderArg,
        /// Candidate pool for hybrid / rerank-ablate (default: config).
        #[arg(long)]
        candidate_pool: Option<usize>,
        /// Load / HybridRerank: enable cross-encoder (requires onnx + model).
        #[arg(long, default_value_t = false)]
        with_rerank: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigAction {
    Check,
}

fn load_config(path: Option<&PathBuf>) -> Result<Config> {
    match path {
        Some(p) => Config::load(Some(p.as_path())).context("failed to load config"),
        None => Config::load_default().context("failed to load default config"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(cli.config.as_ref())?;

    let log_opts = LoggingOptions::from(&config.logging);
    let _ = init_logging(&log_opts);

    match cli.command {
        Commands::Config {
            action: ConfigAction::Check,
        } => {
            config
                .validate()
                .map_err(|e| anyhow::anyhow!("invalid config: {e}"))?;
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
        Commands::Ingest {
            collection,
            path,
            embedder,
        } => {
            run_ingest(&config, &collection, &path, embedder).await?;
        }
        Commands::Delete {
            collection,
            document,
        } => {
            run_delete(&config, &collection, &document).await?;
        }
        Commands::Recover => {
            run_recover(&config).await?;
        }
        Commands::Serve => {
            serve(config)
                .await
                .map_err(|e| anyhow::anyhow!("server failed: {e}"))?;
        }
        Commands::Query {
            collection,
            query,
            top_k,
            mode,
            embedder,
            reranker,
        } => {
            run_query(
                &config,
                &collection,
                &query,
                top_k,
                mode,
                embedder,
                reranker,
            )
            .await?;
        }
        Commands::Ablate {
            collection,
            query,
            top_k,
            candidate_pool,
        } => {
            run_ablate(&config, &collection, &query, top_k, candidate_pool).await?;
        }
        Commands::EmbedParity {
            text,
            python_json,
            query,
        } => {
            let texts = if text.is_empty() {
                vec!["hello world".into()]
            } else {
                text
            };
            run_embed_parity(&config, &texts, python_json.as_deref(), query)?;
        }
        Commands::Bench {
            suite,
            dataset,
            mode,
            top_k,
            requests,
            concurrency,
            out_dir,
            baseline,
            current,
            embedder,
            candidate_pool,
            with_rerank,
        } => {
            run_bench(
                &config,
                &suite,
                &dataset,
                &mode,
                top_k,
                requests,
                concurrency,
                &out_dir,
                baseline.as_deref(),
                current.as_deref(),
                embedder,
                candidate_pool.unwrap_or(config.search.candidate_pool),
                with_rerank,
            )
            .await?;
        }
    }

    Ok(())
}

fn bench_embedder(arg: EmbedderArg) -> BenchEmbedder {
    match arg {
        EmbedderArg::Hash => BenchEmbedder::Hash,
        EmbedderArg::Onnx => BenchEmbedder::Onnx,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_bench(
    config: &Config,
    suite: &str,
    datasets: &[PathBuf],
    mode: &str,
    top_k: usize,
    requests: usize,
    concurrency: usize,
    out_dir: &Path,
    baseline: Option<&Path>,
    current: Option<&Path>,
    embedder_arg: EmbedderArg,
    candidate_pool: usize,
    with_rerank: bool,
) -> Result<()> {
    let embedder = bench_embedder(embedder_arg);
    let dims = match embedder {
        BenchEmbedder::Hash => 32,
        BenchEmbedder::Onnx => config.embedding.dimensions,
    };
    match suite {
        "smoke" => {
            let fixture = PathBuf::from("bench/fixtures/smoke");
            let ds = BeirDataset::load(&fixture).context("load smoke fixture")?;
            let data_dir = config.data_dir.join("bench_smoke");
            let _ = std::fs::remove_dir_all(&data_dir);
            let opts = EvalOptions {
                collection: "smoke".into(),
                mode: parse_mode(mode)?,
                top_k,
                candidate_pool,
                use_cache: false,
                dimensions: 32,
                skip_index: false,
                embedder: BenchEmbedder::Hash,
                onnx_model: None,
                onnx_batch_size: config.embedding.batch_size,
            };
            let report = run_beir_eval(&data_dir, &ds, &opts)
                .await
                .context("smoke eval")?;
            let mut report = report;
            report.suite = "smoke".into();
            let (json, md) = write_eval(out_dir, "smoke", &report)?;
            println!(
                "smoke nDCG@{top_k}={:.4} Recall@{top_k}={:.4} MRR={:.4}",
                report.ndcg_at_k, report.recall_at_k, report.mrr
            );
            println!("wrote {} {}", json.display(), md.display());
            // Soft floor so CI catches total breakage without being brittle.
            if report.ndcg_at_k < 0.3 {
                anyhow::bail!("smoke nDCG@{top_k}={:.4} below floor 0.3", report.ndcg_at_k);
            }
        }
        "beir" => {
            let path = datasets.first().ok_or_else(|| {
                anyhow::anyhow!("--dataset required for suite=beir (see scripts/download_beir.py)")
            })?;
            let ds = BeirDataset::load(path).context("load BEIR dataset")?;
            let data_dir = config.data_dir.join("bench_beir");
            let _ = std::fs::remove_dir_all(&data_dir);
            let opts = EvalOptions {
                collection: "beir".into(),
                mode: parse_mode(mode)?,
                top_k,
                candidate_pool,
                use_cache: false,
                dimensions: dims,
                skip_index: false,
                embedder,
                onnx_model: Some(config.embedding.model_path.clone()),
                onnx_batch_size: config.embedding.batch_size,
            };
            let report = run_beir_eval(&data_dir, &ds, &opts)
                .await
                .context("beir eval")?;
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("beir");
            let (json, md) = write_eval(out_dir, &format!("beir_{name}"), &report)?;
            println!(
                "beir embedder={embedder:?} nDCG@{top_k}={:.4} Recall@{top_k}={:.4} MRR={:.4} queries={}",
                report.ndcg_at_k, report.recall_at_k, report.mrr, report.n_queries
            );
            println!("wrote {} {}", json.display(), md.display());
        }
        "load" => {
            if let Some(path) = datasets.first() {
                let ds = BeirDataset::load(path).context("load BEIR for load-test")?;
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("beir");
                let data_dir = config.data_dir.join(format!("bench_load_{name}"));
                let mode = parse_mode(mode).unwrap_or(retrieva_core::SearchMode::Hybrid);
                let load_opts = LoadOptions {
                    requests,
                    concurrency,
                    mode,
                    embedder,
                    dimensions: dims,
                    onnx_model: Some(config.embedding.model_path.clone()),
                    onnx_batch_size: config.embedding.batch_size,
                    use_cross_encoder: with_rerank,
                    cross_encoder_path: Some(config.rerank.cross_encoder_path.clone()),
                    query_mix: retrieva_bench::QueryMix::Zipf { s: 1.1 },
                };
                let (off, on) = run_beir_load_pair(&data_dir, &ds, &load_opts)
                    .await
                    .context("beir load")?;
                for report in [&off, &on] {
                    let tag = report.suite.clone();
                    let (json, md) = write_load(out_dir, &tag, report)?;
                    println!(
                        "{tag} ok={} err={} p50={:.2}ms p95={:.2}ms p99={:.2}ms qps={:.1} cache={} hit_rate={:.1}% mix={} | {}",
                        report.ok,
                        report.err,
                        report.p50_ms,
                        report.p95_ms,
                        report.p99_ms,
                        report.qps,
                        report.use_cache,
                        report.cache_hit_rate * 100.0,
                        report.query_mix,
                        report.notes
                    );
                    println!("wrote {} {}", json.display(), md.display());
                    if report.err > 0 {
                        anyhow::bail!("load test had {} errors", report.err);
                    }
                }
            } else {
                let data_dir = config.data_dir.join("bench_load");
                let _ = std::fs::remove_dir_all(&data_dir);
                let report =
                    run_load_test(&data_dir, requests, concurrency, "rust bm25 retrieval")
                        .await
                        .context("load test")?;
                let (json, md) = write_load(out_dir, "load_smoke", &report)?;
                println!(
                    "load_smoke (not a capacity claim) ok={} err={} p50={:.2}ms p95={:.2}ms p99={:.2}ms qps={:.1}",
                    report.ok, report.err, report.p50_ms, report.p95_ms, report.p99_ms, report.qps
                );
                println!("wrote {} {}", json.display(), md.display());
                if report.err > 0 {
                    anyhow::bail!("load test had {} errors", report.err);
                }
            }
        }
        "compare" => {
            let baseline = baseline.ok_or_else(|| anyhow::anyhow!("--baseline required"))?;
            let current = current.ok_or_else(|| anyhow::anyhow!("--current required"))?;
            let cmp = compare_eval(baseline, current, 0.02).context("compare")?;
            println!("{}", serde_json::to_string_pretty(&cmp)?);
            if !cmp.ok {
                anyhow::bail!("nDCG regression: delta={:.4} (floor -0.02)", cmp.delta_ndcg);
            }
        }
        "matrix" => {
            let paths: Vec<PathBuf> = if !datasets.is_empty() {
                datasets.to_vec()
            } else if let Ok(list) = std::env::var("RETRIEVA_BENCH_DATASETS") {
                list.split(',')
                    .map(|s| PathBuf::from(s.trim()))
                    .filter(|p| !p.as_os_str().is_empty())
                    .collect()
            } else {
                vec![PathBuf::from("bench/fixtures/smoke")]
            };
            let matrix_opts = MatrixOptions {
                top_k,
                candidate_pool,
                embedder,
                dimensions: dims,
                onnx_model: Some(config.embedding.model_path.clone()),
                onnx_batch_size: config.embedding.batch_size,
                min_hybrid_gain: 0.01,
            };
            let mut all_rows = Vec::new();
            for path in &paths {
                let ds =
                    BeirDataset::load(path).with_context(|| format!("load {}", path.display()))?;
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ds")
                    .to_string();
                let data_dir = config.data_dir.join(format!("bench_matrix_{name}"));
                let rows = run_dataset_matrix(&data_dir, &ds, out_dir, &name, &matrix_opts)
                    .await
                    .with_context(|| format!("matrix {name}"))?;
                all_rows.extend(rows);
            }
            let (hybrid_ok, notes) = assess_hybrid(&all_rows, embedder, 0.01, 0.02);
            let emb_label = match embedder {
                BenchEmbedder::Hash => "hash",
                BenchEmbedder::Onnx => "onnx",
            };
            let report = MatrixReport {
                rows: all_rows,
                hybrid_ok,
                notes,
                embedder: emb_label.into(),
            };
            let path = write_matrix(out_dir, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("wrote {}", path.display());
        }
        "rerank-ablate" => {
            let path = datasets.first().ok_or_else(|| {
                anyhow::anyhow!("--dataset required for suite=rerank-ablate")
            })?;
            let ds = BeirDataset::load(path).context("load BEIR for rerank-ablate")?;
            let data_dir = config.data_dir.join("bench_rerank_ablate");
            let report = run_ce_vs_maxsim_beir(
                &data_dir,
                &ds,
                top_k,
                candidate_pool.max(50),
                embedder,
                dims,
                Some(config.embedding.model_path.clone()),
                config.rerank.cross_encoder_path.clone(),
                config.rerank.maxsim_path.clone(),
            )
            .await
            .context("rerank-ablate")?;
            let out = write_rerank_ablate(out_dir, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("wrote {}", out.display());
        }
        "cache" => {
            let fixture = PathBuf::from("bench/fixtures/smoke");
            let ds = BeirDataset::load(&fixture).context("load smoke")?;
            let mut reports = Vec::new();
            for (label, use_cache) in [("cache_off", false), ("cache_on", true)] {
                let data_dir = config.data_dir.join(format!("bench_{label}"));
                let _ = std::fs::remove_dir_all(&data_dir);
                let opts = EvalOptions {
                    collection: label.into(),
                    mode: parse_mode(mode).unwrap_or(retrieva_core::SearchMode::Hybrid),
                    top_k,
                    candidate_pool,
                    use_cache,
                    dimensions: 32,
                    skip_index: false,
                    embedder: BenchEmbedder::Hash,
                    onnx_model: None,
                    onnx_batch_size: 32,
                };
                let mut report = run_beir_eval(&data_dir, &ds, &opts).await?;
                if use_cache {
                    let mut opts2 = opts.clone();
                    opts2.skip_index = true;
                    report = run_beir_eval(&data_dir, &ds, &opts2).await?;
                }
                report.suite = label.into();
                write_eval(out_dir, label, &report)?;
                reports.push(report);
            }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        other => {
            anyhow::bail!(
                "unknown suite '{other}' (expected smoke|beir|load|compare|matrix|cache|rerank-ablate)"
            )
        }
    }
    Ok(())
}

fn run_embed_parity(
    config: &Config,
    texts: &[String],
    python_json: Option<&Path>,
    query: bool,
) -> Result<()> {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (config, texts, python_json, query);
        anyhow::bail!(
            "embed-parity requires `--features onnx` and models from scripts/download_bge_small.py"
        );
    }
    #[cfg(feature = "onnx")]
    {
        use retrieva_ingest::OnnxEmbedder;
        let emb = OnnxEmbedder::from_config(&config.embedding)
            .context("load ONNX")?
            .with_query_mode(query);
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let rust_vecs: Vec<Vec<f32>> = if query {
            refs.iter()
                .map(|t| emb.embed_query_sync(t).map(|e| e.values))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            emb.embed_documents_sync(&refs)?
                .into_iter()
                .map(|e| e.values)
                .collect()
        };
        let out = serde_json::json!({
            "texts": texts,
            "query": query,
            "embeddings": rust_vecs,
            "batch_size": texts.len(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);

        if let Some(py_path) = python_json {
            let raw = std::fs::read_to_string(py_path)?;
            let py: serde_json::Value = serde_json::from_str(&raw)?;
            let py_list = py["embeddings"]
                .as_array()
                .context("python JSON missing embeddings")?;
            if py_list.len() != rust_vecs.len() {
                anyhow::bail!(
                    "batch size mismatch python={} rust={}",
                    py_list.len(),
                    rust_vecs.len()
                );
            }
            let mut min_cos = 1.0f32;
            for (i, (rust_vec, py_emb)) in rust_vecs.iter().zip(py_list.iter()).enumerate() {
                let py_vec: Vec<f32> = py_emb
                    .as_array()
                    .context("embedding row")?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                if py_vec.len() != rust_vec.len() {
                    anyhow::bail!(
                        "dim mismatch at [{i}] python={} rust={}",
                        py_vec.len(),
                        rust_vec.len()
                    );
                }
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for (a, b) in rust_vec.iter().zip(py_vec.iter()) {
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let cos = dot / (na.sqrt() * nb.sqrt());
                println!("cosine_vs_python[{i}]={cos:.6} chars={}", texts[i].len());
                min_cos = min_cos.min(cos);
            }
            println!("cosine_vs_python_min={min_cos:.6} n={}", rust_vecs.len());
            if min_cos < 0.999 {
                anyhow::bail!(
                    "parity failed: min cosine {min_cos:.6} < 0.999 (check mean-pool / prefix / padding)"
                );
            }
        }
        Ok(())
    }
}

async fn run_ingest(
    config: &Config,
    collection: &str,
    path: &Path,
    embedder_kind: EmbedderArg,
) -> Result<()> {
    let collection_id = CollectionId::new(collection);
    std::fs::create_dir_all(&config.data_dir)
        .with_context(|| format!("create data_dir {}", config.data_dir.display()))?;

    let batch = ingest_path(path, &collection_id, &config.ingest).context("ingest failed")?;
    let index = IncrementalIndex::open(&config.data_dir, config.embedding.dimensions)
        .await
        .context("open incremental index")?;

    index
        .upsert_documents(&batch.documents)
        .await
        .context("store documents")?;

    let dense_items = embed_chunks(config, embedder_kind, &batch.chunks).await?;
    let vectors: Vec<Vec<f32>> = dense_items.into_iter().map(|(_, v)| v).collect();
    index
        .upsert_chunks(&collection_id, &batch.chunks, Some(&vectors))
        .await
        .context("WAL upsert chunks")?;

    info!(
        collection = %collection_id,
        files_ok = batch.files_ok,
        files_skipped = batch.files_skipped,
        documents = batch.documents.len(),
        chunks = batch.chunks.len(),
        embedder = ?embedder_kind,
        "ingest complete"
    );

    println!(
        "ingested collection={collection} files_ok={} files_skipped={} documents={} chunks={} embedder={embedder_kind:?}",
        batch.files_ok,
        batch.files_skipped,
        batch.documents.len(),
        batch.chunks.len()
    );
    Ok(())
}

async fn run_delete(config: &Config, collection: &str, document: &str) -> Result<()> {
    let index = IncrementalIndex::open(&config.data_dir, config.embedding.dimensions)
        .await
        .context("open incremental index")?;
    index
        .delete_document(&CollectionId::new(collection), &DocumentId::new(document))
        .await
        .context("delete document")?;
    println!("deleted collection={collection} document={document}");
    Ok(())
}

async fn run_recover(config: &Config) -> Result<()> {
    let index = IncrementalIndex::open(&config.data_dir, config.embedding.dimensions)
        .await
        .context("open + recover")?;
    info!(data_dir = %index.data_dir().display(), "WAL recovery finished");
    println!("recovered data_dir={}", index.data_dir().display());
    Ok(())
}

async fn run_query(
    config: &Config,
    collection: &str,
    query: &str,
    top_k: usize,
    mode: QueryModeArg,
    embedder_kind: EmbedderArg,
    reranker_arg: RerankerArg,
) -> Result<()> {
    if matches!(embedder_kind, EmbedderArg::Onnx) {
        #[cfg(not(feature = "onnx"))]
        {
            anyhow::bail!(
                "ONNX embedder requires `--features onnx` (rustc >= 1.92). Or use `--embedder hash`."
            );
        }
    }

    let search_mode = match mode {
        QueryModeArg::Bm25 => SearchMode::Bm25,
        QueryModeArg::Dense => SearchMode::Dense,
        QueryModeArg::Hybrid => SearchMode::Hybrid,
        QueryModeArg::HybridRerank => SearchMode::HybridRerank,
    };

    let request = SearchRequest {
        collection_id: CollectionId::new(collection),
        query: query.to_string(),
        top_k,
        mode: search_mode,
        filters: SearchFilters::default(),
        rerank_top_n: config.rerank.top_n,
        use_cache: true,
    };

    let reranker = resolve_reranker(config, reranker_arg)?;

    let response = match embedder_kind {
        EmbedderArg::Hash => {
            let engine = build_hash_engine(config, reranker)?;
            engine.search(request).await.context("search failed")?
        }
        EmbedderArg::Onnx => {
            #[cfg(feature = "onnx")]
            {
                use retrieva_ingest::OnnxEmbedder;
                let sparse = Arc::new(TantivySparseIndex::open(&config.data_dir)?);
                let vectors = Arc::new(UsearchVectorIndex::open(
                    &config.data_dir,
                    config.embedding.dimensions,
                )?);
                let emb = Arc::new(
                    OnnxEmbedder::from_config(&config.embedding)
                        .context("load ONNX embedder")?
                        .with_query_mode(true),
                );
                let cache = cache_from_config(&config.cache, Arc::clone(&emb));
                let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&config.search)?);
                let engine = HybridEngine::new(
                    sparse,
                    vectors,
                    emb,
                    fusion,
                    reranker,
                    config.search.candidate_pool,
                )
                .with_cache(cache);
                engine.search(request).await.context("search failed")?
            }
            #[cfg(not(feature = "onnx"))]
            {
                unreachable!("gated above");
            }
        }
    };

    print_hits(&response, mode);
    Ok(())
}

async fn run_ablate(
    config: &Config,
    collection: &str,
    query: &str,
    top_k: usize,
    candidate_pool: usize,
) -> Result<()> {
    let sparse = TantivySparseIndex::open(&config.data_dir).context("open BM25")?;
    let vectors = UsearchVectorIndex::open(&config.data_dir, config.embedding.dimensions)
        .context("open ANN")?;
    let emb = HashEmbedder::new(config.embedding.dimensions);
    let fusion = fusion_from_config(&config.search)?;

    let collection_id = CollectionId::new(collection);
    let bm25 = sparse
        .search(&collection_id, query, candidate_pool)
        .await
        .context("bm25")?;
    let q_emb = emb.embed(query).await.context("embed")?;
    let dense = vectors
        .search_chunks(&collection_id, &q_emb.values, candidate_pool)
        .await
        .context("dense")?;
    let candidates = fusion
        .fuse(&[bm25, dense], candidate_pool)
        .context("fuse")?;

    // Prefer ColBERT ONNX MaxSim + CE when models exist. Hash MaxSim is not a research arm.
    let mut arms: Vec<(String, Arc<dyn Reranker>)> = Vec::new();
    extend_ablation_arms(config, &mut arms);
    let has_onnx_maxsim = arms.iter().any(|(n, _)| n == "maxsim_onnx");
    let has_ce = arms.iter().any(|(n, _)| n == "cross_encoder");
    if !has_onnx_maxsim {
        arms.insert(0, ("maxsim_hash".into(), Arc::new(HashMaxSimReranker::new(32))));
        warn!("maxsim_onnx missing; using hash MaxSim (not a ColBERT quality claim)");
    }
    if arms.len() < 2 {
        arms.insert(0, ("identity".into(), Arc::new(IdentityReranker)));
        warn!("fell back to identity wiring check; install CE + ColBERT ONNX for real ablation");
    }
    if has_ce && has_onnx_maxsim {
        arms.retain(|(n, _)| n == "cross_encoder" || n == "maxsim_onnx");
    }

    let report = ablate_rerankers(query, &candidates, top_k, &arms)
        .await
        .context("ablate")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn extend_ablation_arms(config: &Config, arms: &mut Vec<(String, Arc<dyn Reranker>)>) {
    #[cfg(feature = "onnx")]
    {
        if config.rerank.cross_encoder_path.exists() {
            match retrieva_rerank::CrossEncoderReranker::from_config(&config.rerank) {
                Ok(ce) => arms.push(("cross_encoder".into(), Arc::new(ce))),
                Err(e) => warn!(error = %e, "skipping cross_encoder arm"),
            }
        }
        if config.rerank.maxsim_path.exists() {
            match retrieva_rerank::OnnxMaxSimReranker::from_config(&config.rerank) {
                Ok(ms) => arms.push(("maxsim_onnx".into(), Arc::new(ms))),
                Err(e) => warn!(error = %e, "skipping maxsim_onnx arm"),
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (config, arms);
    }
}

fn resolve_reranker(config: &Config, arg: RerankerArg) -> Result<Arc<dyn Reranker>> {
    match arg {
        RerankerArg::Config => reranker_from_config(&config.rerank).context("reranker_from_config"),
        RerankerArg::Identity => Ok(Arc::new(IdentityReranker)),
        RerankerArg::Maxsim => Ok(Arc::new(HashMaxSimReranker::new(32))),
        RerankerArg::CrossEncoder => {
            let mut cfg = config.rerank.clone();
            cfg.backend = "cross_encoder".into();
            reranker_from_config(&cfg).context("cross_encoder")
        }
    }
}

fn build_hash_engine(
    config: &Config,
    reranker: Arc<dyn Reranker>,
) -> Result<HybridEngine<HashEmbedder>> {
    let sparse = Arc::new(TantivySparseIndex::open(&config.data_dir)?);
    let vectors = Arc::new(UsearchVectorIndex::open(
        &config.data_dir,
        config.embedding.dimensions,
    )?);
    let emb = Arc::new(HashEmbedder::new(config.embedding.dimensions));
    let cache = cache_from_config(&config.cache, Arc::clone(&emb));
    let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&config.search)?);
    Ok(HybridEngine::new(
        sparse,
        vectors,
        emb,
        fusion,
        reranker,
        config.search.candidate_pool,
    )
    .with_cache(cache))
}

fn print_hits(response: &retrieva_core::SearchResponse, mode: QueryModeArg) {
    println!(
        "mode={mode:?} cache_hit={} hits={} cache_ms={:.2} bm25_ms={:.2} dense_ms={:.2} fusion_ms={:.2} rerank_ms={:.2} embed_ms={:.2} total_ms={:.2}",
        response.cache_hit,
        response.results.len(),
        response.latency.cache_ms,
        response.latency.bm25_ms,
        response.latency.dense_ms,
        response.latency.fusion_ms,
        response.latency.rerank_ms,
        response.latency.embed_ms,
        response.latency.total_ms
    );
    for (rank, hit) in response.results.iter().enumerate() {
        let preview: String = hit.chunk.text.chars().take(160).collect();
        println!(
            "{:>3}. score={:.4} id={} doc={}\n     {}",
            rank + 1,
            hit.scores.final_score,
            hit.chunk.id,
            hit.chunk.document_id,
            preview
        );
    }
}

async fn embed_chunks(
    config: &Config,
    kind: EmbedderArg,
    chunks: &[retrieva_core::Chunk],
) -> Result<Vec<(retrieva_core::Chunk, Vec<f32>)>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    match kind {
        EmbedderArg::Hash => {
            let emb = HashEmbedder::new(config.embedding.dimensions);
            let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
            let vectors = emb.embed_batch(&texts).await.context("hash embed")?;
            Ok(chunks
                .iter()
                .zip(vectors)
                .map(|(c, e)| (c.clone(), e.values))
                .collect())
        }
        EmbedderArg::Onnx => {
            #[cfg(feature = "onnx")]
            {
                use retrieva_ingest::OnnxEmbedder;
                let emb = OnnxEmbedder::from_config(&config.embedding)
                    .context("load ONNX embedder")?
                    .with_query_mode(false);
                let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
                let vectors = emb
                    .embed_documents_sync(&texts)
                    .context("onnx embed documents")?;
                Ok(chunks
                    .iter()
                    .zip(vectors)
                    .map(|(c, e)| (c.clone(), e.values))
                    .collect())
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = config;
                warn!("onnx feature disabled; refusing --embedder onnx");
                anyhow::bail!(
                    "ONNX embedder requires `--features onnx` (rustc >= 1.92). \
                     Download weights with scripts/download_bge_small.py"
                );
            }
        }
    }
}
