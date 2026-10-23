//! Incremental index: WAL → DocStore + BM25 + ANN, with crash recovery.

use crate::store::FsDocStore;
use crate::vector::UsearchVectorIndex;
use crate::wal::{WalOp, WriteAheadLog};
use crate::TantivySparseIndex;
use chrono::Utc;
use parking_lot::Mutex;
use retrieva_core::traits::DocStore;
use retrieva_core::{
    Chunk, ChunkId, CollectionId, CollectionInfo, Document, DocumentId, Result, RetrievaError,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Coordinates durable upsert/delete across store, sparse, and dense indexes.
pub struct IncrementalIndex {
    data_dir: PathBuf,
    wal: Mutex<WriteAheadLog>,
    store: Arc<FsDocStore>,
    sparse: Arc<TantivySparseIndex>,
    vectors: Arc<UsearchVectorIndex>,
}

impl IncrementalIndex {
    /// Open indexes under `data_dir` and replay any uncheckpointed WAL entries.
    pub async fn open(data_dir: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let wal = WriteAheadLog::open(&data_dir)?;
        let store = Arc::new(FsDocStore::open(&data_dir)?);
        let sparse = Arc::new(TantivySparseIndex::open(&data_dir)?);
        let vectors = Arc::new(UsearchVectorIndex::open(&data_dir, dimensions)?);

        let engine = Self {
            data_dir,
            wal: Mutex::new(wal),
            store,
            sparse,
            vectors,
        };
        engine.recover().await?;
        Ok(engine)
    }

    /// Data directory root.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Shared document store.
    pub fn store(&self) -> Arc<FsDocStore> {
        Arc::clone(&self.store)
    }

    /// Shared BM25 index.
    pub fn sparse(&self) -> Arc<TantivySparseIndex> {
        Arc::clone(&self.sparse)
    }

    /// Shared ANN index.
    pub fn vectors(&self) -> Arc<UsearchVectorIndex> {
        Arc::clone(&self.vectors)
    }

    /// Ensure a collection row exists in the doc store.
    pub async fn ensure_collection(&self, id: &CollectionId, name: &str) -> Result<()> {
        if self.store.get_collection(id).await?.is_none() {
            self.store
                .create_collection(&CollectionInfo {
                    id: id.clone(),
                    name: name.to_string(),
                    created_at: Utc::now(),
                    chunk_count: 0,
                })
                .await?;
        }
        Ok(())
    }

    /// Persist documents metadata (no WAL — documents are recovered via chunk ops).
    pub async fn upsert_documents(&self, docs: &[Document]) -> Result<()> {
        self.store.upsert_documents(docs).await
    }

    /// WAL-backed upsert of chunks (+ optional dense vectors aligned 1:1).
    pub async fn upsert_chunks(
        &self,
        collection_id: &CollectionId,
        chunks: &[Chunk],
        vectors: Option<&[Vec<f32>]>,
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        if let Some(vecs) = vectors {
            if vecs.len() != chunks.len() {
                return Err(RetrievaError::InvalidArgument(format!(
                    "vectors len {} != chunks len {}",
                    vecs.len(),
                    chunks.len()
                )));
            }
        }

        self.ensure_collection(collection_id, collection_id.as_str())
            .await?;

        let op = WalOp::UpsertChunks {
            collection_id: collection_id.clone(),
            chunks: chunks.to_vec(),
            vectors: vectors.map(|v| v.to_vec()),
        };
        let seq = {
            let mut wal = self.wal.lock();
            wal.append(op)?
        };
        self.apply_upsert(collection_id, chunks, vectors).await?;
        self.wal.lock().checkpoint(seq)?;
        Ok(())
    }

    /// WAL-backed delete of a document (all chunks) from store + indexes.
    pub async fn delete_document(
        &self,
        collection_id: &CollectionId,
        document_id: &DocumentId,
    ) -> Result<()> {
        let op = WalOp::DeleteDocument {
            collection_id: collection_id.clone(),
            document_id: document_id.clone(),
        };
        let seq = {
            let mut wal = self.wal.lock();
            wal.append(op)?
        };
        self.apply_delete_document(collection_id, document_id)
            .await?;
        self.wal.lock().checkpoint(seq)?;
        Ok(())
    }

    /// WAL-backed delete of a single chunk.
    pub async fn delete_chunk(
        &self,
        collection_id: &CollectionId,
        chunk_id: &ChunkId,
    ) -> Result<()> {
        let op = WalOp::DeleteChunk {
            collection_id: collection_id.clone(),
            chunk_id: chunk_id.as_str().to_string(),
        };
        let seq = {
            let mut wal = self.wal.lock();
            wal.append(op)?
        };
        self.apply_delete_chunk(collection_id, chunk_id).await?;
        self.wal.lock().checkpoint(seq)?;
        Ok(())
    }

    /// Replay WAL entries after the last checkpoint (crash recovery).
    pub async fn recover(&self) -> Result<()> {
        let applied = self.wal.lock().applied_seq()?;
        let pending = self.wal.lock().entries_after(applied)?;
        if pending.is_empty() {
            return Ok(());
        }
        info!(count = pending.len(), after = applied, "replaying WAL");
        let mut last = applied;
        for entry in pending {
            match &entry.op {
                WalOp::UpsertChunks {
                    collection_id,
                    chunks,
                    vectors,
                } => {
                    let vecs = vectors.as_deref();
                    self.apply_upsert(collection_id, chunks, vecs).await?;
                }
                WalOp::DeleteDocument {
                    collection_id,
                    document_id,
                } => {
                    self.apply_delete_document(collection_id, document_id)
                        .await?;
                }
                WalOp::DeleteChunk {
                    collection_id,
                    chunk_id,
                } => {
                    self.apply_delete_chunk(collection_id, &ChunkId::new(chunk_id))
                        .await?;
                }
            }
            last = entry.seq;
        }
        self.wal.lock().checkpoint(last)?;
        info!(applied_seq = last, "WAL recovery complete");
        Ok(())
    }

    async fn apply_upsert(
        &self,
        collection_id: &CollectionId,
        chunks: &[Chunk],
        vectors: Option<&[Vec<f32>]>,
    ) -> Result<()> {
        self.store.put_chunks(chunks).await?;
        self.sparse.upsert_batch(chunks).await?;
        if let Some(vecs) = vectors {
            let items: Vec<(Chunk, Vec<f32>)> =
                chunks.iter().cloned().zip(vecs.iter().cloned()).collect();
            self.vectors.upsert_batch(&items).await?;
        } else {
            warn!(
                collection = %collection_id,
                "upsert without dense vectors; ANN not updated"
            );
        }
        Ok(())
    }

    async fn apply_delete_document(
        &self,
        collection_id: &CollectionId,
        document_id: &DocumentId,
    ) -> Result<()> {
        let chunk_ids = self.store.list_chunks_for_document(document_id).await?;
        for cid in &chunk_ids {
            self.vectors
                .remove_chunk(collection_id, cid.as_str())
                .await?;
        }
        self.sparse
            .remove_document(collection_id, document_id)
            .await?;
        self.store.delete_document(document_id).await?;
        Ok(())
    }

    async fn apply_delete_chunk(
        &self,
        collection_id: &CollectionId,
        chunk_id: &ChunkId,
    ) -> Result<()> {
        self.vectors
            .remove_chunk(collection_id, chunk_id.as_str())
            .await?;
        self.sparse
            .remove_in_collection(collection_id, chunk_id)
            .await?;
        self.store.delete_chunk(chunk_id).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::traits::SparseIndex;
    use std::collections::HashMap;

    fn chunk(col: &str, doc: &str, id: &str, text: &str) -> Chunk {
        Chunk {
            id: ChunkId::new(id),
            document_id: DocumentId::new(doc),
            collection_id: CollectionId::new(col),
            text: text.into(),
            ordinal: 0,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn upsert_search_delete_roundtrip() {
        let dir = tempfile::tempdir().expect("temp");
        let dims = 8usize;
        let eng = IncrementalIndex::open(dir.path(), dims)
            .await
            .expect("open");
        let col = CollectionId::new("c");
        let c1 = chunk("c", "d1", "d1#0", "rust retrieval bm25");
        let v1 = vec![1.0f32; dims];
        eng.upsert_chunks(&col, std::slice::from_ref(&c1), Some(&[v1]))
            .await
            .expect("upsert");

        let hits = eng.sparse().search(&col, "rust", 5).await.expect("search");
        assert_eq!(hits.len(), 1);

        eng.delete_document(&col, &DocumentId::new("d1"))
            .await
            .expect("delete");
        let hits2 = eng.sparse().search(&col, "rust", 5).await.expect("search2");
        assert!(hits2.is_empty());
        assert!(eng
            .store()
            .get_chunk(&ChunkId::new("d1#0"))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn crash_recovery_replays_wal() {
        let dir = tempfile::tempdir().expect("temp");
        let dims = 8usize;
        let path = dir.path().to_path_buf();

        // Simulate: WAL append without applying / checkpoint.
        {
            let mut wal = WriteAheadLog::open(&path).expect("wal");
            let c = chunk("c", "d1", "d1#0", "recover me please");
            wal.append(WalOp::UpsertChunks {
                collection_id: CollectionId::new("c"),
                chunks: vec![c],
                vectors: Some(vec![vec![0.5f32; dims]]),
            })
            .expect("append");
            // No checkpoint — crash.
        }

        let eng = IncrementalIndex::open(&path, dims)
            .await
            .expect("recover open");
        let hits = eng
            .sparse()
            .search(&CollectionId::new("c"), "recover", 5)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id.as_str(), "d1#0");
        assert_eq!(eng.wal.lock().applied_seq().unwrap(), 1);
    }
}
