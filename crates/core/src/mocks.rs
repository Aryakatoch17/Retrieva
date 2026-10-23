//! In-memory mock implementations of core traits for unit tests.

use crate::traits::{
    DocStore, Embedder, Fusion, QueryCache, Reranker, Retriever, SparseIndex, VectorIndex,
};
use crate::types::{
    Chunk, ChunkId, CollectionId, CollectionInfo, Document, DocumentId, Embedding,
    LatencyBreakdown, Neighbor, ScoreBreakdown, ScoredChunk, SearchMode, SearchRequest,
    SearchResponse, Stage,
};
use crate::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

/// Deterministic hash-based embedder (no ONNX). Useful for cache / fusion tests.
#[derive(Debug, Default)]
pub struct MockEmbedder {
    /// Fixed dimensionality.
    pub dims: usize,
}

impl MockEmbedder {
    /// Create a mock embedder with the given dimensionality.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    fn embed_sync(&self, text: &str) -> Embedding {
        let mut values = vec![0.0_f32; self.dims];
        for (i, byte) in text.bytes().enumerate() {
            let idx = i % self.dims;
            values[idx] += f32::from(byte) / 255.0;
        }
        // L2 normalize
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut values {
            *v /= norm;
        }
        Embedding::new(values)
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.embed_sync(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|t| self.embed_sync(t)).collect())
    }
}

/// In-memory brute-force vector index.
#[derive(Debug, Default)]
pub struct MockVectorIndex {
    vectors: RwLock<HashMap<u64, Vec<f32>>>,
}

impl MockVectorIndex {
    /// Create an empty mock vector index.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorIndex for MockVectorIndex {
    async fn upsert(&self, key: u64, vector: &[f32]) -> Result<()> {
        let mut guard = self
            .vectors
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.insert(key, vector.to_vec());
        Ok(())
    }

    async fn remove(&self, key: u64) -> Result<()> {
        let mut guard = self
            .vectors
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.remove(&key);
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Neighbor>> {
        let guard = self
            .vectors
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        let mut scored: Vec<Neighbor> = guard
            .iter()
            .map(|(key, vec)| {
                let score = cosine(query, vec);
                Neighbor { key: *key, score }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn len(&self) -> Result<usize> {
        let guard = self
            .vectors
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        Ok(guard.len())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// In-memory sparse index with naive term overlap scoring.
#[derive(Debug, Default)]
pub struct MockSparseIndex {
    chunks: RwLock<HashMap<String, Chunk>>,
}

impl MockSparseIndex {
    /// Create an empty mock sparse index.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SparseIndex for MockSparseIndex {
    async fn upsert(&self, chunk: &Chunk) -> Result<()> {
        let mut guard = self
            .chunks
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.insert(chunk.id.as_str().to_string(), chunk.clone());
        Ok(())
    }

    async fn remove(&self, chunk_id: &ChunkId) -> Result<()> {
        let mut guard = self
            .chunks
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.remove(chunk_id.as_str());
        Ok(())
    }

    async fn search(
        &self,
        collection_id: &CollectionId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        let guard = self
            .chunks
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        let q_terms: Vec<&str> = query.split_whitespace().collect();
        let mut hits: Vec<ScoredChunk> = guard
            .values()
            .filter(|c| c.collection_id == *collection_id)
            .filter_map(|chunk| {
                let text_lower = chunk.text.to_lowercase();
                let score = q_terms
                    .iter()
                    .filter(|t| text_lower.contains(&t.to_lowercase()))
                    .count() as f32;
                if score <= 0.0 {
                    return None;
                }
                Some(ScoredChunk {
                    chunk: chunk.clone(),
                    scores: ScoreBreakdown {
                        bm25: Some(score),
                        final_score: score,
                        ..ScoreBreakdown::default()
                    },
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.scores
                .final_score
                .partial_cmp(&a.scores.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}

/// In-memory document store.
#[derive(Debug, Default)]
pub struct MockDocStore {
    collections: RwLock<HashMap<String, CollectionInfo>>,
    documents: RwLock<HashMap<String, Document>>,
    chunks: RwLock<HashMap<String, Chunk>>,
}

impl MockDocStore {
    /// Create an empty mock doc store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl DocStore for MockDocStore {
    async fn create_collection(&self, info: &CollectionInfo) -> Result<()> {
        let mut guard = self
            .collections
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.insert(info.id.as_str().to_string(), info.clone());
        Ok(())
    }

    async fn delete_collection(&self, id: &CollectionId) -> Result<()> {
        let mut collections = self
            .collections
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        collections.remove(id.as_str());
        Ok(())
    }

    async fn get_collection(&self, id: &CollectionId) -> Result<Option<CollectionInfo>> {
        let guard = self
            .collections
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        Ok(guard.get(id.as_str()).cloned())
    }

    async fn upsert_documents(&self, docs: &[Document]) -> Result<()> {
        let mut guard = self
            .documents
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        for doc in docs {
            guard.insert(doc.id.as_str().to_string(), doc.clone());
        }
        Ok(())
    }

    async fn delete_document(&self, id: &DocumentId) -> Result<()> {
        let mut docs = self
            .documents
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        docs.remove(id.as_str());
        let mut chunks = self
            .chunks
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        chunks.retain(|_, c| c.document_id != *id);
        Ok(())
    }

    async fn get_chunk(&self, id: &ChunkId) -> Result<Option<Chunk>> {
        let guard = self
            .chunks
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        Ok(guard.get(id.as_str()).cloned())
    }

    async fn put_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let mut guard = self
            .chunks
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        for chunk in chunks {
            guard.insert(chunk.id.as_str().to_string(), chunk.clone());
        }
        Ok(())
    }

    async fn list_chunks_for_document(&self, id: &DocumentId) -> Result<Vec<ChunkId>> {
        let guard = self
            .chunks
            .read()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        Ok(guard
            .values()
            .filter(|c| c.document_id == *id)
            .map(|c| c.id.clone())
            .collect())
    }

    async fn delete_chunk(&self, id: &ChunkId) -> Result<()> {
        let mut guard = self
            .chunks
            .write()
            .map_err(|e| crate::RetrievaError::internal(e.to_string()))?;
        guard.remove(id.as_str());
        Ok(())
    }
}

/// Pass-through reranker that sorts by existing `final_score`.
#[derive(Debug, Default)]
pub struct MockReranker;

#[async_trait]
impl Reranker for MockReranker {
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

/// Reciprocal Rank Fusion mock (k=60). Real impl lives in `retrieva-fusion`.
#[derive(Debug, Clone)]
pub struct MockRrfFusion {
    /// RRF constant.
    pub k: u32,
}

impl Default for MockRrfFusion {
    fn default() -> Self {
        Self { k: 60 }
    }
}

impl Fusion for MockRrfFusion {
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
                if let Some(bm25) = hit.scores.bm25 {
                    entry.0.scores.bm25 = Some(bm25);
                }
                if let Some(dense) = hit.scores.dense {
                    entry.0.scores.dense = Some(dense);
                }
            }
        }
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
        Ok(fused)
    }
}

/// No-op cache.
#[derive(Debug, Default)]
pub struct MockQueryCache;

#[async_trait]
impl QueryCache for MockQueryCache {
    async fn get_with_query_embedding(
        &self,
        _request: &SearchRequest,
        _query_embedding: Option<&[f32]>,
    ) -> Result<Option<SearchResponse>> {
        Ok(None)
    }

    async fn put_with_query_embedding(
        &self,
        _request: &SearchRequest,
        _response: &SearchResponse,
        _query_embedding: Option<&[f32]>,
    ) -> Result<()> {
        Ok(())
    }

    async fn invalidate_collection(&self, _collection_id: &CollectionId) -> Result<()> {
        Ok(())
    }
}

/// Simple BM25-only retriever over [`MockSparseIndex`].
#[derive(Debug)]
pub struct MockRetriever {
    /// Sparse backend.
    pub sparse: MockSparseIndex,
}

impl MockRetriever {
    /// Create a retriever backed by a new empty sparse index.
    pub fn new() -> Self {
        Self {
            sparse: MockSparseIndex::new(),
        }
    }
}

impl Default for MockRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Retriever for MockRetriever {
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse> {
        request.validate()?;
        let start = Instant::now();
        let mut latency = LatencyBreakdown::default();

        let bm25_start = Instant::now();
        let mut results = self
            .sparse
            .search(&request.collection_id, &request.query, request.top_k)
            .await?;
        latency.record(Stage::Bm25, bm25_start.elapsed());

        if matches!(
            request.mode,
            SearchMode::Bm25 | SearchMode::Hybrid | SearchMode::HybridRerank
        ) {
            // Mock only implements sparse; dense/rerank stages stay zeroed.
        }

        results.truncate(request.top_k);
        latency.record(Stage::Total, start.elapsed());

        Ok(SearchResponse {
            results,
            latency,
            cache_hit: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Embedder, SparseIndex, VectorIndex};

    #[tokio::test]
    async fn mock_embedder_is_deterministic() {
        let emb = MockEmbedder::new(8);
        let a = emb.embed("hello").await.expect("embed");
        let b = emb.embed("hello").await.expect("embed");
        assert_eq!(a, b);
        assert_eq!(a.dim(), 8);
    }

    #[tokio::test]
    async fn mock_vector_index_roundtrip() {
        let idx = MockVectorIndex::new();
        idx.upsert(1, &[1.0, 0.0, 0.0]).await.expect("upsert");
        idx.upsert(2, &[0.0, 1.0, 0.0]).await.expect("upsert");
        let hits = idx.search(&[1.0, 0.0, 0.0], 1).await.expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key, 1);
    }

    #[tokio::test]
    async fn mock_sparse_finds_overlap() {
        let idx = MockSparseIndex::new();
        let chunk = Chunk {
            id: ChunkId::new("c1"),
            document_id: DocumentId::new("d1"),
            collection_id: CollectionId::new("col"),
            text: "rust retrieval engine".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        idx.upsert(&chunk).await.expect("upsert");
        let hits = idx
            .search(&CollectionId::new("col"), "rust engine", 5)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].scores.bm25.unwrap() >= 2.0);
    }

    #[tokio::test]
    async fn mock_retriever_validates_empty_query() {
        let r = MockRetriever::new();
        let req = SearchRequest {
            collection_id: CollectionId::new("col"),
            query: "  ".into(),
            top_k: 10,
            mode: SearchMode::Bm25,
            filters: Default::default(),
            rerank_top_n: 50,
            use_cache: false,
        };
        let err = r.search(req).await.expect_err("empty query");
        assert!(matches!(err, crate::RetrievaError::InvalidArgument(_)));
    }
}
