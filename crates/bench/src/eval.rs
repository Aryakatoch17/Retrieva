//! End-to-end BEIR evaluation against [`HybridEngine`].

use crate::beir::BeirDataset;
use crate::metrics::{mean_mrr, mean_ndcg_at_k, mean_recall_at_k};
use crate::report::EvalReport;
use anyhow::{Context, Result};
use retrieva_core::traits::{Embedder, Fusion, Retriever};
use retrieva_core::{
    Chunk, ChunkId, CollectionId, Config, Document, DocumentId, SearchFilters, SearchMode,
    SearchRequest,
};
use retrieva_index::IncrementalIndex;
use retrieva_ingest::{chunk_text, HashEmbedder};
use retrieva_server::HybridEngine;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

/// Which embedder backs indexing + query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BenchEmbedder {
    /// Deterministic hash — **not** a hybrid quality claim.
    #[default]
    Hash,
    /// ONNX BGE (or configured model) — real dense / hybrid story.
    Onnx,
}

impl BenchEmbedder {
    /// Parse `hash` / `onnx`.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "hash" => Ok(Self::Hash),
            "onnx" => Ok(Self::Onnx),
            other => anyhow::bail!("unknown embedder '{other}' (hash|onnx)"),
        }
    }
}

/// Options for a BEIR eval run.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    /// Collection id to index into.
    pub collection: String,
    /// Search mode.
    pub mode: SearchMode,
    /// Ranking cutoff.
    pub top_k: usize,
    /// Candidate pool for hybrid.
    pub candidate_pool: usize,
    /// Consult / fill the query cache.
    pub use_cache: bool,
    /// Embedding dimensionality (hash or ONNX).
    pub dimensions: usize,
    /// When true, open an existing index and skip upsert (matrix modes after the first).
    pub skip_index: bool,
    /// Embedder backend.
    pub embedder: BenchEmbedder,
    /// ONNX model path (required when `embedder == Onnx`).
    pub onnx_model: Option<PathBuf>,
    /// Batch size for ONNX document embedding.
    pub onnx_batch_size: usize,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            collection: "beir".into(),
            mode: SearchMode::Bm25,
            top_k: 10,
            candidate_pool: 50,
            use_cache: false,
            dimensions: 32,
            skip_index: false,
            embedder: BenchEmbedder::Hash,
            onnx_model: None,
            onnx_batch_size: 32,
        }
    }
}

/// Index a BEIR corpus into a fresh data dir and evaluate all queries.
pub async fn run_beir_eval(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &EvalOptions,
) -> Result<EvalReport> {
    match opts.embedder {
        BenchEmbedder::Hash => run_beir_eval_hash(data_dir, dataset, opts).await,
        BenchEmbedder::Onnx => run_beir_eval_onnx(data_dir, dataset, opts).await,
    }
}

async fn run_beir_eval_hash(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &EvalOptions,
) -> Result<EvalReport> {
    std::fs::create_dir_all(data_dir)?;
    let dims = opts.dimensions.max(8);
    let index = IncrementalIndex::open(data_dir, dims)
        .await
        .context("open incremental index")?;
    let collection = CollectionId::new(&opts.collection);

    if !opts.skip_index {
        let emb = HashEmbedder::new(dims);
        index_corpus_sync(&index, &collection, dataset, Arc::new(emb)).await?;
    }

    let engine = build_hash_engine(&index, dims, opts.candidate_pool, opts.use_cache)?;
    evaluate_queries(&engine, dataset, opts, &collection).await
}

async fn run_beir_eval_onnx(
    data_dir: &Path,
    dataset: &BeirDataset,
    opts: &EvalOptions,
) -> Result<EvalReport> {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (data_dir, dataset, opts);
        anyhow::bail!("ONNX embedder requires `--features onnx` on retrieva-cli / retrieva-bench");
    }
    #[cfg(feature = "onnx")]
    {
        use retrieva_core::EmbeddingConfig;
        use retrieva_ingest::OnnxEmbedder;

        std::fs::create_dir_all(data_dir)?;
        let dims = opts.dimensions.max(8);
        let model = opts
            .onnx_model
            .clone()
            .unwrap_or_else(|| PathBuf::from("models/bge-small-en-v1.5/model.onnx"));
        let index = IncrementalIndex::open(data_dir, dims)
            .await
            .context("open incremental index")?;
        let collection = CollectionId::new(&opts.collection);

        let emb_cfg = EmbeddingConfig {
            model_path: model,
            model_id: "onnx".into(),
            dimensions: dims,
            batch_size: opts.onnx_batch_size.max(1),
        };
        // One ORT session for index + query (process allows only one Environment).
        let emb = Arc::new(
            OnnxEmbedder::from_config(&emb_cfg)
                .context("load OnnxEmbedder")?
                .with_query_mode(false),
        );
        if !opts.skip_index {
            index_corpus_sync(&index, &collection, dataset, Arc::clone(&emb)).await?;
        }
        emb.set_query_mode(true);
        let engine = build_onnx_engine(&index, emb, opts.candidate_pool, opts.use_cache)?;
        evaluate_queries(&engine, dataset, opts, &collection).await
    }
}

async fn index_corpus_sync<E: Embedder + Send + Sync + 'static>(
    index: &IncrementalIndex,
    collection: &CollectionId,
    dataset: &BeirDataset,
    emb: Arc<E>,
) -> Result<()> {
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
        let parts = chunk_text(&text, 512, 64);
        for (ord, part) in parts.into_iter().enumerate() {
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
    let vectors: Vec<Vec<f32>> = emb
        .embed_batch(&texts)
        .await
        .context("embed corpus")?
        .into_iter()
        .map(|e| e.values)
        .collect();
    index
        .upsert_chunks(collection, &chunks, Some(&vectors))
        .await?;
    info!(
        docs = documents.len(),
        chunks = chunks.len(),
        "indexed BEIR corpus"
    );
    Ok(())
}

async fn evaluate_queries<E: Embedder + Send + Sync + 'static>(
    engine: &HybridEngine<E>,
    dataset: &BeirDataset,
    opts: &EvalOptions,
    collection: &CollectionId,
) -> Result<EvalReport> {
    let mut rankings: HashMap<String, Vec<String>> = HashMap::new();
    let t0 = Instant::now();
    let mut n_queries = 0usize;

    for (qid, q) in &dataset.queries {
        if !dataset.qrels.contains_key(qid) {
            continue;
        }
        n_queries += 1;
        let req = SearchRequest {
            collection_id: collection.clone(),
            query: q.text.clone(),
            top_k: opts.top_k,
            mode: opts.mode,
            filters: SearchFilters::default(),
            rerank_top_n: opts.candidate_pool.max(opts.top_k),
            use_cache: opts.use_cache,
        };
        let resp = engine.search(req).await.context("search")?;
        let mut seen = HashMap::new();
        let mut docs = Vec::new();
        for hit in resp.results {
            let did = hit.chunk.document_id.as_str().to_string();
            if seen.insert(did.clone(), ()).is_none() {
                docs.push(did);
            }
        }
        rankings.insert(qid.clone(), docs);
    }
    let search_ms_total = t0.elapsed().as_secs_f64() * 1000.0;

    let ndcg = mean_ndcg_at_k(&dataset.qrels, &rankings, opts.top_k);
    let recall = mean_recall_at_k(&dataset.qrels, &rankings, opts.top_k);
    let mrr = mean_mrr(&dataset.qrels, &rankings);

    Ok(EvalReport {
        suite: "beir".into(),
        dataset: dataset.root.display().to_string(),
        mode: format!("{:?}", opts.mode),
        top_k: opts.top_k,
        n_queries,
        ndcg_at_k: ndcg,
        recall_at_k: recall,
        mrr,
        search_ms_total,
        created_at: String::new(),
    }
    .stamp())
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

    let mut cfg = Config::default();
    cfg.search.fusion = "weighted".into();
    cfg.search.bm25_weight = 0.65;
    cfg.search.dense_weight = 0.35;
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
    candidate_pool: usize,
    use_cache: bool,
) -> Result<HybridEngine<retrieva_ingest::OnnxEmbedder>> {
    use retrieva_cache::cache_from_config;
    use retrieva_fusion::fusion_from_config;
    use retrieva_rerank::IdentityReranker;

    // Equal weights once dense is a real encoder.
    // HybridRerank quality belongs in rerank-ablate (real CE / ColBERT), not hash MaxSim.
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
        Arc::new(IdentityReranker),
        candidate_pool,
    );
    if use_cache {
        engine = engine.with_cache(cache_from_config(&cfg.cache, emb));
    }
    Ok(engine)
}

/// Resolve search mode from a CLI string.
pub fn parse_mode(s: &str) -> Result<SearchMode> {
    match s.to_ascii_lowercase().as_str() {
        "bm25" => Ok(SearchMode::Bm25),
        "dense" => Ok(SearchMode::Dense),
        "hybrid" => Ok(SearchMode::Hybrid),
        "hybrid+rerank" | "hybrid_rerank" => Ok(SearchMode::HybridRerank),
        other => anyhow::bail!("unknown mode '{other}'"),
    }
}
