//! Concurrent query load tests (smoke corpus or BEIR).

use crate::beir::BeirDataset;
use crate::eval::BenchEmbedder;
use crate::report::{percentile, LoadReport};
use anyhow::{Context, Result};
use retrieva_core::traits::{Embedder, Fusion, Retriever};
use retrieva_core::{
    Chunk, ChunkId, CollectionId, Config, Document, DocumentId, EmbeddingConfig, SearchFilters,
    SearchMode, SearchRequest,
};
use retrieva_index::IncrementalIndex;
use retrieva_ingest::{chunk_text, HashEmbedder};
use retrieva_server::HybridEngine;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::info;

/// Host label for load-report footnotes (best-effort).
pub fn host_fingerprint() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    format!("{os}/{arch} logical_cpus={cpus}")
}

/// How load-test requests sample from the query pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryMix {
    /// `queries[i % n]` — unique under small request counts → ~0% cache hits.
    RoundRobin,
    /// Zipf(s) over the sorted query list (realistic repeated head).
    Zipf {
        /// Exponent (1.1 ≈ mild head skew).
        s: f64,
    },
}

impl QueryMix {
    fn label(self) -> &'static str {
        match self {
            QueryMix::RoundRobin => "round_robin",
            QueryMix::Zipf { .. } => "zipf",
        }
    }
}

/// Load-test options (hash smoke vs real ONNX path).
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Requests.
    pub requests: usize,
    /// Concurrency.
    pub concurrency: usize,
    /// Search mode.
    pub mode: SearchMode,
    /// Embedder.
    pub embedder: BenchEmbedder,
    /// Dims.
    pub dimensions: usize,
    /// ONNX model.
    pub onnx_model: Option<PathBuf>,
    /// ONNX batch.
    pub onnx_batch_size: usize,
    /// When true, use config cross-encoder for HybridRerank (onnx feature).
    pub use_cross_encoder: bool,
    /// Cross-encoder model path.
    pub cross_encoder_path: Option<PathBuf>,
    /// Query sampling (default Zipf so cache-on is meaningful).
    pub query_mix: QueryMix,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            requests: 50,
            concurrency: 4,
            mode: SearchMode::Hybrid,
            embedder: BenchEmbedder::Hash,
            dimensions: 32,
            onnx_model: None,
            onnx_batch_size: 32,
            use_cross_encoder: false,
            cross_encoder_path: None,
            query_mix: QueryMix::Zipf { s: 1.1 },
        }
    }
}

/// Tiny in-process smoke load (not a capacity claim).
pub async fn run_load_test(
    data_dir: &Path,
    requests: usize,
    concurrency: usize,
    query: &str,
) -> Result<LoadReport> {
    let dims = 32usize;
    let collection = CollectionId::new("load");
    let index = IncrementalIndex::open(data_dir, dims).await?;
    let chunk = Chunk {
        id: ChunkId::new("d1#0"),
        document_id: DocumentId::new("d1"),
        collection_id: collection.clone(),
        text: "hybrid retrieval load test document about rust bm25 dense vectors".into(),
        ordinal: 0,
        metadata: HashMap::new(),
    };
    let emb = HashEmbedder::new(dims);
    let vector = emb.embed(&chunk.text).await?.values;
    index
        .upsert_chunks(&collection, std::slice::from_ref(&chunk), Some(&[vector]))
        .await?;

    let engine = Arc::new(build_hash_engine(&index, dims, 50, false)?);
    fire_load_generic(
        engine,
        &collection,
        &[query.to_string()],
        requests,
        concurrency,
        SearchMode::Bm25,
        false,
        QueryMix::RoundRobin,
        "load_smoke",
        "smoke (1 doc)",
    )
    .await
}

/// Index a BEIR corpus once, then measure QPS for cache-off and cache-on passes.
pub async fn run_beir_load_pair(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &LoadOptions,
) -> Result<(LoadReport, LoadReport)> {
    match opts.embedder {
        BenchEmbedder::Hash => run_beir_load_pair_hash(data_dir, dataset, opts).await,
        BenchEmbedder::Onnx => run_beir_load_pair_onnx(data_dir, dataset, opts).await,
    }
}

async fn run_beir_load_pair_hash(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &LoadOptions,
) -> Result<(LoadReport, LoadReport)> {
    let dims = opts.dimensions.max(8);
    let collection = CollectionId::new("beir_load");
    let _ = std::fs::remove_dir_all(data_dir);
    let index = IncrementalIndex::open(data_dir, dims)
        .await
        .context("open index")?;

    let mut documents = Vec::new();
    let mut chunks = Vec::new();
    for (doc_id, doc) in &dataset.corpus {
        let text = doc.full_text();
        documents.push(Document {
            id: DocumentId::new(doc_id),
            collection_id: collection.clone(),
            text: text.clone(),
            metadata: HashMap::new(),
        });
        for (ord, part) in chunk_text(&text, 512, 64).into_iter().enumerate() {
            chunks.push(Chunk {
                id: ChunkId::new(format!("{doc_id}#{ord}")),
                document_id: DocumentId::new(doc_id),
                collection_id: collection.clone(),
                text: part,
                ordinal: ord as u32,
                metadata: HashMap::new(),
            });
        }
    }
    index.upsert_documents(&documents).await?;
    let emb = HashEmbedder::new(dims);
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors: Vec<Vec<f32>> = emb
        .embed_batch(&texts)
        .await?
        .into_iter()
        .map(|e| e.values)
        .collect();
    index
        .upsert_chunks(&collection, &chunks, Some(&vectors))
        .await?;
    info!(
        docs = documents.len(),
        chunks = chunks.len(),
        "beir load corpus indexed"
    );

    let mut queries: Vec<String> = dataset.queries.values().map(|q| q.text.clone()).collect();
    queries.sort();
    if queries.is_empty() {
        anyhow::bail!("dataset has no queries");
    }

    let name = dataset
        .root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("beir");
    let ds_note = format!(
        "BEIR {name}; embedder=hash; docs={}; chunks={}; queries_pool={}; mode={:?}; mix={}; {}",
        documents.len(),
        chunks.len(),
        queries.len(),
        opts.mode,
        opts.query_mix.label(),
        host_fingerprint()
    );

    let engine_off = Arc::new(build_hash_engine(&index, dims, 50, false)?);
    let off = fire_load_generic(
        engine_off,
        &collection,
        &queries,
        opts.requests,
        opts.concurrency,
        opts.mode,
        false,
        opts.query_mix,
        &format!("load_{name}_cache_off"),
        &ds_note,
    )
    .await?;
    let engine_on = Arc::new(build_hash_engine(&index, dims, 50, true)?);
    if let Some(c) = engine_on.cache() {
        c.reset_stats();
    }
    let on = fire_load_generic(
        engine_on,
        &collection,
        &queries,
        opts.requests,
        opts.concurrency,
        opts.mode,
        true,
        opts.query_mix,
        &format!("load_{name}_cache_on"),
        &ds_note,
    )
    .await?;
    Ok((off, on))
}

async fn run_beir_load_pair_onnx(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &LoadOptions,
) -> Result<(LoadReport, LoadReport)> {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (data_dir, dataset, opts);
        anyhow::bail!("ONNX load requires `--features onnx`");
    }
    #[cfg(feature = "onnx")]
    {
        use retrieva_ingest::OnnxEmbedder;
        use retrieva_rerank::{CrossEncoderReranker, IdentityReranker};

        let dims = opts.dimensions.max(8);
        let model = opts
            .onnx_model
            .clone()
            .unwrap_or_else(|| PathBuf::from("models/bge-small-en-v1.5/model.onnx"));
        let collection = CollectionId::new("beir_load");
        let _ = std::fs::remove_dir_all(data_dir);
        let index = IncrementalIndex::open(data_dir, dims)
            .await
            .context("open index")?;

        let emb_cfg = EmbeddingConfig {
            model_path: model.clone(),
            model_id: "BAAI/bge-small-en-v1.5".into(),
            dimensions: dims,
            batch_size: opts.onnx_batch_size.max(1),
        };
        let doc_emb = Arc::new(
            OnnxEmbedder::from_config(&emb_cfg)
                .context("onnx embedder")?
                .with_query_mode(false),
        );

        let mut documents = Vec::new();
        let mut chunks = Vec::new();
        for (doc_id, doc) in &dataset.corpus {
            let text = doc.full_text();
            documents.push(Document {
                id: DocumentId::new(doc_id),
                collection_id: collection.clone(),
                text: text.clone(),
                metadata: HashMap::new(),
            });
            for (ord, part) in chunk_text(&text, 512, 64).into_iter().enumerate() {
                chunks.push(Chunk {
                    id: ChunkId::new(format!("{doc_id}#{ord}")),
                    document_id: DocumentId::new(doc_id),
                    collection_id: collection.clone(),
                    text: part,
                    ordinal: ord as u32,
                    metadata: HashMap::new(),
                });
            }
        }
        index.upsert_documents(&documents).await?;
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let vectors: Vec<Vec<f32>> = doc_emb
            .embed_batch(&texts)
            .await?
            .into_iter()
            .map(|e| e.values)
            .collect();
        index
            .upsert_chunks(&collection, &chunks, Some(&vectors))
            .await?;

        let mut queries: Vec<String> = dataset.queries.values().map(|q| q.text.clone()).collect();
        queries.sort();
        if queries.is_empty() {
            anyhow::bail!("dataset has no queries");
        }

        let name = dataset
            .root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("beir");
        let rerank_note = if opts.use_cross_encoder {
            "rerank=cross_encoder"
        } else {
            "rerank=identity"
        };
        let ds_note = format!(
            "BEIR {name}; embedder=onnx/bge-small; docs={}; chunks={}; queries_pool={}; mode={:?}; {rerank_note}; mix={}; {}",
            documents.len(),
            chunks.len(),
            queries.len(),
            opts.mode,
            opts.query_mix.label(),
            host_fingerprint()
        );

        doc_emb.set_query_mode(true);
        let reranker: Arc<dyn retrieva_core::traits::Reranker> = if opts.use_cross_encoder {
            let path = opts
                .cross_encoder_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("models/bge-reranker-base/model.onnx"));
            let mut cfg = Config::default().rerank;
            cfg.cross_encoder_path = path;
            cfg.backend = "cross_encoder".into();
            Arc::new(CrossEncoderReranker::from_config(&cfg).context("cross_encoder")?)
        } else {
            Arc::new(IdentityReranker)
        };

        let engine_off = Arc::new(build_onnx_engine(
            &index,
            Arc::clone(&doc_emb),
            Arc::clone(&reranker),
            50,
            false,
        )?);
        let off = fire_load_generic(
            engine_off,
            &collection,
            &queries,
            opts.requests,
            opts.concurrency,
            opts.mode,
            false,
            opts.query_mix,
            &format!("load_{name}_onnx_cache_off"),
            &ds_note,
        )
        .await?;
        let engine_on = Arc::new(build_onnx_engine(
            &index,
            doc_emb,
            reranker,
            50,
            true,
        )?);
        if let Some(c) = engine_on.cache() {
            c.reset_stats();
        }
        let on = fire_load_generic(
            engine_on,
            &collection,
            &queries,
            opts.requests,
            opts.concurrency,
            opts.mode,
            true,
            opts.query_mix,
            &format!("load_{name}_onnx_cache_on"),
            &ds_note,
        )
        .await?;
        Ok((off, on))
    }
}

#[allow(clippy::too_many_arguments)]
async fn fire_load_generic<E: Embedder + Send + Sync + 'static>(
    engine: Arc<HybridEngine<E>>,
    collection: &CollectionId,
    queries: &[String],
    requests: usize,
    concurrency: usize,
    mode: SearchMode,
    use_cache: bool,
    query_mix: QueryMix,
    suite: &str,
    notes: &str,
) -> Result<LoadReport> {
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(requests);
    let picker = QueryPicker::new(queries, query_mix);

    for i in 0..requests {
        let permit = Arc::clone(&sem).acquire_owned().await?;
        let engine = Arc::clone(&engine);
        let collection = collection.clone();
        let q = picker.pick(i).to_string();
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            let req = SearchRequest {
                collection_id: collection,
                query: q,
                top_k: 10,
                mode,
                filters: SearchFilters::default(),
                rerank_top_n: 50,
                use_cache,
            };
            match engine.search(req).await {
                Ok(resp) => (true, resp.cache_hit, started.elapsed()),
                Err(_) => (false, false, started.elapsed()),
            }
        }));
    }

    let mut latencies = Vec::with_capacity(requests);
    let mut ok = 0usize;
    let mut err = 0usize;
    let mut cache_hits = 0u64;
    let mut cache_misses = 0u64;
    for h in handles {
        let (success, hit, dur) = h.await.context("join")?;
        if success {
            ok += 1;
            if use_cache {
                if hit {
                    cache_hits += 1;
                } else {
                    cache_misses += 1;
                }
            }
        } else {
            err += 1;
        }
        latencies.push(dur.as_secs_f64() * 1000.0);
    }
    let wall = t0.elapsed();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let qps = if wall.as_secs_f64() > 0.0 {
        requests as f64 / wall.as_secs_f64()
    } else {
        0.0
    };
    let denom = cache_hits + cache_misses;
    let cache_hit_rate = if denom == 0 {
        0.0
    } else {
        cache_hits as f64 / denom as f64
    };

    if let Some(c) = engine.cache() {
        let s = c.stats();
        info!(
            suite,
            ok,
            err,
            qps,
            use_cache,
            cache_hit_rate,
            layer_hits = s.hits,
            layer_misses = s.misses,
            mix = query_mix.label(),
            "load test complete"
        );
    } else {
        info!(
            suite,
            ok,
            err,
            qps,
            use_cache,
            cache_hit_rate,
            mix = query_mix.label(),
            "load test complete"
        );
    }

    Ok(LoadReport {
        suite: suite.into(),
        requests,
        concurrency: concurrency.max(1),
        ok,
        err,
        p50_ms: percentile(&latencies, 50.0),
        p95_ms: percentile(&latencies, 95.0),
        p99_ms: percentile(&latencies, 99.0),
        mean_ms: mean,
        qps,
        use_cache,
        cache_hits,
        cache_misses,
        cache_hit_rate,
        query_mix: query_mix.label().into(),
        notes: notes.into(),
        created_at: String::new(),
    }
    .stamp())
}

struct QueryPicker<'a> {
    queries: &'a [String],
    mix: QueryMix,
    zipf_cdf: Vec<f64>,
}

impl<'a> QueryPicker<'a> {
    fn new(queries: &'a [String], mix: QueryMix) -> Self {
        let zipf_cdf = match mix {
            QueryMix::Zipf { s } => zipf_cdf(queries.len().max(1), s),
            QueryMix::RoundRobin => Vec::new(),
        };
        Self {
            queries,
            mix,
            zipf_cdf,
        }
    }

    fn pick(&self, request_i: usize) -> &str {
        if self.queries.is_empty() {
            return "";
        }
        let idx = match self.mix {
            QueryMix::RoundRobin => request_i % self.queries.len(),
            QueryMix::Zipf { .. } => {
                let u = unit_hash(request_i as u64);
                self.zipf_cdf
                    .partition_point(|&x| x < u)
                    .min(self.queries.len() - 1)
            }
        };
        self.queries[idx].as_str()
    }
}

fn zipf_cdf(n: usize, s: f64) -> Vec<f64> {
    let s = if s.is_finite() && s > 0.0 { s } else { 1.1 };
    let mut weights: Vec<f64> = (1..=n).map(|k| 1.0 / (k as f64).powf(s)).collect();
    let sum: f64 = weights.iter().sum::<f64>().max(1e-12);
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for w in &mut weights {
        *w /= sum;
        acc += *w;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn unit_hash(mut x: u64) -> f64 {
    // splitmix64 → (0, 1]
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z as f64) / (u64::MAX as f64)
}

fn build_hash_engine(
    index: &IncrementalIndex,
    dims: usize,
    candidate_pool: usize,
    use_cache: bool,
) -> Result<HybridEngine<HashEmbedder>> {
    use retrieva_cache::cache_from_config;
    use retrieva_fusion::fusion_from_config;
    use retrieva_rerank::IdentityReranker;

    let cfg = Config::default();
    let emb = Arc::new(HashEmbedder::new(dims));
    let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&cfg.search)?);
    let mut engine = HybridEngine::new(
        index.sparse(),
        index.vectors(),
        Arc::clone(&emb),
        fusion,
        Arc::new(IdentityReranker),
        candidate_pool,
    );
    if use_cache {
        engine = engine.with_cache(cache_from_config(&cfg.cache, emb));
    }
    Ok(engine)
}

#[cfg(feature = "onnx")]
fn build_onnx_engine(
    index: &IncrementalIndex,
    emb: Arc<retrieva_ingest::OnnxEmbedder>,
    reranker: Arc<dyn retrieva_core::traits::Reranker>,
    candidate_pool: usize,
    use_cache: bool,
) -> Result<HybridEngine<retrieva_ingest::OnnxEmbedder>> {
    use retrieva_cache::cache_from_config;
    use retrieva_fusion::fusion_from_config;

    let mut cfg = Config::default();
    cfg.search.fusion = "weighted".into();
    cfg.search.bm25_weight = 0.5;
    cfg.search.dense_weight = 0.5;
    let fusion: Arc<dyn Fusion> = Arc::from(fusion_from_config(&cfg.search)?);
    let mut engine = HybridEngine::new(
        index.sparse(),
        index.vectors(),
        Arc::clone(&emb),
        fusion,
        reranker,
        candidate_pool,
    );
    if use_cache {
        engine = engine.with_cache(cache_from_config(&cfg.cache, emb));
    }
    Ok(engine)
}
