//! usearch-backed [`VectorIndex`] with per-collection persistence + chunk sidecar.

use crate::keys::chunk_key;
use async_trait::async_trait;
use parking_lot::Mutex;
use retrieva_core::traits::VectorIndex;
use retrieva_core::{
    Chunk, CollectionId, Neighbor, Result, RetrievaError, ScoreBreakdown, ScoredChunk,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::debug;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// On-disk ANN index rooted at `data_dir/indexes/<collection>/usearch`.
pub struct UsearchVectorIndex {
    root: PathBuf,
    dimensions: usize,
    open: Mutex<HashMap<String, Arc<Mutex<CollectionAnn>>>>,
}

struct CollectionAnn {
    index: Index,
    index_path: PathBuf,
    chunks: HashMap<u64, Chunk>,
    chunks_path: PathBuf,
}

impl UsearchVectorIndex {
    /// Open (or create) the vector index store under `data_dir`.
    pub fn open(data_dir: impl Into<PathBuf>, dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(RetrievaError::InvalidArgument(
                "vector dimensions must be >= 1".into(),
            ));
        }
        let root = data_dir.into().join("indexes");
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            dimensions,
            open: Mutex::new(HashMap::new()),
        })
    }

    /// Embedding dimensionality this store was opened with.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Upsert many chunk vectors with a single save per collection.
    pub async fn upsert_batch(&self, items: &[(Chunk, Vec<f32>)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        let mut by_collection: HashMap<String, Vec<(&Chunk, &[f32])>> = HashMap::new();
        for (chunk, vec) in items {
            by_collection
                .entry(chunk.collection_id.as_str().to_string())
                .or_default()
                .push((chunk, vec.as_slice()));
        }

        for (collection, group) in by_collection {
            let handle = self.get_or_open(&collection)?;
            let mut guard = handle.lock();
            for (chunk, vector) in group {
                Self::upsert_sync(&mut guard, chunk, vector, self.dimensions)?;
            }
            Self::persist(&guard)?;
            debug!(collection = %collection, "saved usearch batch");
        }
        Ok(())
    }

    /// Remove a chunk vector by chunk id within a collection.
    pub async fn remove_chunk(&self, collection_id: &CollectionId, chunk_id: &str) -> Result<()> {
        let handle = self.get_or_open(collection_id.as_str())?;
        let mut guard = handle.lock();
        let key = chunk_key(chunk_id);
        if guard.index.contains(key) {
            let _ = guard
                .index
                .remove(key)
                .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        }
        guard.chunks.remove(&key);
        Self::persist(&guard)?;
        Ok(())
    }

    /// Search a collection and resolve hits back to [`ScoredChunk`]s.
    pub async fn search_chunks(
        &self,
        collection_id: &CollectionId,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        if top_k == 0 {
            return Err(RetrievaError::InvalidArgument("top_k must be >= 1".into()));
        }
        if query.len() != self.dimensions {
            return Err(RetrievaError::InvalidArgument(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dimensions
            )));
        }

        let dir = self.collection_dir(collection_id.as_str());
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let handle = self.get_or_open(collection_id.as_str())?;
        let guard = handle.lock();
        let matches = guard
            .index
            .search(query, top_k)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;

        let mut hits = Vec::new();
        for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
            let Some(chunk) = guard.chunks.get(key) else {
                continue;
            };
            // usearch Cos metric returns distance; convert to similarity-like score.
            let score = 1.0_f32 - *distance;
            hits.push(ScoredChunk {
                chunk: chunk.clone(),
                scores: ScoreBreakdown {
                    dense: Some(score),
                    final_score: score,
                    ..ScoreBreakdown::default()
                },
            });
        }
        Ok(hits)
    }

    fn collection_dir(&self, collection_id: &str) -> PathBuf {
        self.root.join(collection_id).join("usearch")
    }

    fn get_or_open(&self, collection_id: &str) -> Result<Arc<Mutex<CollectionAnn>>> {
        {
            let guard = self.open.lock();
            if let Some(existing) = guard.get(collection_id) {
                return Ok(Arc::clone(existing));
            }
        }

        let dir = self.collection_dir(collection_id);
        std::fs::create_dir_all(&dir)?;
        let opened = open_collection(&dir, self.dimensions)?;
        let arc = Arc::new(Mutex::new(opened));
        let mut guard = self.open.lock();
        Ok(Arc::clone(
            guard.entry(collection_id.to_string()).or_insert(arc),
        ))
    }

    fn upsert_sync(
        col: &mut CollectionAnn,
        chunk: &Chunk,
        vector: &[f32],
        dimensions: usize,
    ) -> Result<()> {
        if vector.len() != dimensions {
            return Err(RetrievaError::InvalidArgument(format!(
                "vector dim {} != index dim {dimensions}",
                vector.len()
            )));
        }
        let key = chunk_key(chunk.id.as_str());
        if col.index.contains(key) {
            let _ = col
                .index
                .remove(key)
                .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        }
        // Grow capacity as needed.
        let needed = col.index.size() + 1;
        if needed > col.index.capacity() {
            col.index
                .reserve(needed.max(16))
                .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        }
        col.index
            .add(key, vector)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        col.chunks.insert(key, chunk.clone());
        Ok(())
    }

    fn persist(col: &CollectionAnn) -> Result<()> {
        col.index
            .save(
                col.index_path
                    .to_str()
                    .ok_or_else(|| RetrievaError::VectorIndex("non-utf8 usearch path".into()))?,
            )
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        let entries: HashMap<String, Chunk> = col
            .chunks
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        std::fs::write(&col.chunks_path, json)?;
        Ok(())
    }
}

fn open_collection(dir: &Path, dimensions: usize) -> Result<CollectionAnn> {
    let index_path = dir.join("index.usearch");
    let chunks_path = dir.join("chunks.json");

    let index = if index_path.exists() {
        Index::restore(
            index_path
                .to_str()
                .ok_or_else(|| RetrievaError::VectorIndex("non-utf8 path".into()))?,
        )
        .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?
    } else {
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..IndexOptions::default()
        };
        let index = Index::new(&options).map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        index
            .reserve(64)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        index
    };

    if index.dimensions() != dimensions {
        return Err(RetrievaError::VectorIndex(format!(
            "on-disk usearch dims {} != configured {dimensions}",
            index.dimensions()
        )));
    }

    let chunks: HashMap<u64, Chunk> = if chunks_path.exists() {
        let raw = std::fs::read_to_string(&chunks_path)?;
        let entries: HashMap<String, Chunk> =
            serde_json::from_str(&raw).map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        entries
            .into_iter()
            .map(|(k, v)| {
                k.parse::<u64>()
                    .map(|key| (key, v))
                    .map_err(|e| RetrievaError::VectorIndex(format!("bad chunk key {k}: {e}")))
            })
            .collect::<Result<HashMap<_, _>>>()?
    } else {
        HashMap::new()
    };

    Ok(CollectionAnn {
        index,
        index_path,
        chunks,
        chunks_path,
    })
}

/// Trait adapter bound to a single collection (for mock-style call sites).
pub struct CollectionVectorIndex {
    store: Arc<UsearchVectorIndex>,
    collection_id: CollectionId,
}

impl CollectionVectorIndex {
    /// Bind `store` to `collection_id`.
    pub fn new(store: Arc<UsearchVectorIndex>, collection_id: CollectionId) -> Self {
        Self {
            store,
            collection_id,
        }
    }
}

#[async_trait]
impl VectorIndex for CollectionVectorIndex {
    async fn upsert(&self, key: u64, vector: &[f32]) -> Result<()> {
        // Trait path without a full Chunk — store a placeholder chunk.
        let chunk = Chunk {
            id: retrieva_core::ChunkId::new(format!("key-{key}")),
            document_id: retrieva_core::DocumentId::new("unknown"),
            collection_id: self.collection_id.clone(),
            text: String::new(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        // Ensure key matches chunk_key of id when possible; for raw keys use direct add.
        let handle = self.store.get_or_open(self.collection_id.as_str())?;
        let mut guard = handle.lock();
        if vector.len() != self.store.dimensions() {
            return Err(RetrievaError::InvalidArgument(format!(
                "vector dim {} != index dim {}",
                vector.len(),
                self.store.dimensions()
            )));
        }
        if guard.index.contains(key) {
            let _ = guard
                .index
                .remove(key)
                .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        }
        let needed = guard.index.size() + 1;
        if needed > guard.index.capacity() {
            guard
                .index
                .reserve(needed.max(16))
                .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        }
        guard
            .index
            .add(key, vector)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        guard.chunks.insert(key, chunk);
        UsearchVectorIndex::persist(&guard)?;
        Ok(())
    }

    async fn remove(&self, key: u64) -> Result<()> {
        let handle = self.store.get_or_open(self.collection_id.as_str())?;
        let mut guard = handle.lock();
        let _ = guard
            .index
            .remove(key)
            .map_err(|e| RetrievaError::VectorIndex(e.to_string()))?;
        guard.chunks.remove(&key);
        UsearchVectorIndex::persist(&guard)?;
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Neighbor>> {
        let hits = self
            .store
            .search_chunks(&self.collection_id, query, top_k)
            .await?;
        Ok(hits
            .into_iter()
            .map(|h| Neighbor {
                key: chunk_key(h.chunk.id.as_str()),
                score: h.scores.final_score,
            })
            .collect())
    }

    async fn len(&self) -> Result<usize> {
        let dir = self.store.collection_dir(self.collection_id.as_str());
        if !dir.exists() {
            return Ok(0);
        }
        let handle = self.store.get_or_open(self.collection_id.as_str())?;
        let guard = handle.lock();
        Ok(guard.index.size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::traits::Embedder;
    use retrieva_core::{ChunkId, DocumentId};
    use retrieva_ingest::HashEmbedder;

    #[tokio::test]
    async fn dense_roundtrip_with_hash_embedder() {
        let dir = tempfile::tempdir().expect("tempdir");
        let emb = HashEmbedder::new(32);
        let store = UsearchVectorIndex::open(dir.path(), 32).expect("open");

        let chunk = Chunk {
            id: ChunkId::new("c1"),
            document_id: DocumentId::new("d1"),
            collection_id: CollectionId::new("col"),
            text: "rust retrieval hybrid engine".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        let other = Chunk {
            id: ChunkId::new("c2"),
            document_id: DocumentId::new("d2"),
            collection_id: CollectionId::new("col"),
            text: "baking sourdough bread recipes".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };

        let v1 = emb.embed(&chunk.text).await.expect("e1");
        let v2 = emb.embed(&other.text).await.expect("e2");
        store
            .upsert_batch(&[(chunk.clone(), v1.values), (other, v2.values)])
            .await
            .expect("upsert");

        let q = emb.embed("rust hybrid retrieval").await.expect("q");
        let hits = store
            .search_chunks(&CollectionId::new("col"), &q.values, 2)
            .await
            .expect("search");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk.id.as_str(), "c1");
    }
}
