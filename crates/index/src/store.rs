//! Filesystem JSON [`DocStore`] (Phase 6). RocksDB can replace this later.

use async_trait::async_trait;
use parking_lot::Mutex;
use retrieva_core::traits::DocStore;
use retrieva_core::{
    Chunk, ChunkId, CollectionId, CollectionInfo, Document, DocumentId, Result, RetrievaError,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// On-disk document / chunk metadata under `data_dir/store/`.
pub struct FsDocStore {
    root: PathBuf,
    // Process-local mutex so concurrent writers don't clobber JSON files.
    lock: Arc<Mutex<()>>,
}

impl FsDocStore {
    /// Open (or create) the store directory.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let root = data_dir.as_ref().join("store");
        std::fs::create_dir_all(root.join("collections"))?;
        std::fs::create_dir_all(root.join("documents"))?;
        std::fs::create_dir_all(root.join("chunks"))?;
        std::fs::create_dir_all(root.join("doc_chunks"))?;
        Ok(Self {
            root,
            lock: Arc::new(Mutex::new(())),
        })
    }

    fn collection_path(&self, id: &CollectionId) -> PathBuf {
        self.root
            .join("collections")
            .join(format!("{}.json", id.as_str()))
    }

    fn document_path(&self, id: &DocumentId) -> PathBuf {
        // Documents may share ids across collections rarely; include collection in path via id only
        // (caller ids are expected unique). Escape path separators.
        let safe = id.as_str().replace(['/', '\\'], "_");
        self.root.join("documents").join(format!("{safe}.json"))
    }

    fn chunk_path(&self, id: &ChunkId) -> PathBuf {
        let safe = id.as_str().replace(['/', '\\', '#'], "_");
        self.root.join("chunks").join(format!("{safe}.json"))
    }

    fn doc_chunks_path(&self, id: &DocumentId) -> PathBuf {
        let safe = id.as_str().replace(['/', '\\'], "_");
        self.root.join("doc_chunks").join(format!("{safe}.json"))
    }

    fn write_json<T: serde::Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(value)
            .map_err(|e| RetrievaError::Storage(format!("encode: {e}")))?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn read_json<T: serde::de::DeserializeOwned>(&self, path: &Path) -> Result<Option<T>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path)?;
        let v = serde_json::from_str(&raw)
            .map_err(|e| RetrievaError::Storage(format!("decode {}: {e}", path.display())))?;
        Ok(Some(v))
    }
}

#[async_trait]
impl DocStore for FsDocStore {
    async fn create_collection(&self, info: &CollectionInfo) -> Result<()> {
        let _g = self.lock.lock();
        self.write_json(&self.collection_path(&info.id), info)
    }

    async fn delete_collection(&self, id: &CollectionId) -> Result<()> {
        let _g = self.lock.lock();
        let path = self.collection_path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        // Best-effort: leave orphan docs/chunks; incremental deletes should run first.
        Ok(())
    }

    async fn get_collection(&self, id: &CollectionId) -> Result<Option<CollectionInfo>> {
        let _g = self.lock.lock();
        self.read_json(&self.collection_path(id))
    }

    async fn upsert_documents(&self, docs: &[Document]) -> Result<()> {
        let _g = self.lock.lock();
        for doc in docs {
            self.write_json(&self.document_path(&doc.id), doc)?;
        }
        Ok(())
    }

    async fn delete_document(&self, id: &DocumentId) -> Result<()> {
        let _g = self.lock.lock();
        let chunk_ids: Vec<ChunkId> = self
            .read_json::<Vec<String>>(&self.doc_chunks_path(id))?
            .unwrap_or_default()
            .into_iter()
            .map(ChunkId::new)
            .collect();
        for cid in &chunk_ids {
            let p = self.chunk_path(cid);
            if p.exists() {
                std::fs::remove_file(p)?;
            }
        }
        let dp = self.document_path(id);
        if dp.exists() {
            std::fs::remove_file(dp)?;
        }
        let dcp = self.doc_chunks_path(id);
        if dcp.exists() {
            std::fs::remove_file(dcp)?;
        }
        Ok(())
    }

    async fn get_chunk(&self, id: &ChunkId) -> Result<Option<Chunk>> {
        let _g = self.lock.lock();
        self.read_json(&self.chunk_path(id))
    }

    async fn put_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let _g = self.lock.lock();
        let mut by_doc: HashMap<String, Vec<String>> = HashMap::new();
        for chunk in chunks {
            self.write_json(&self.chunk_path(&chunk.id), chunk)?;
            by_doc
                .entry(chunk.document_id.as_str().to_string())
                .or_default()
                .push(chunk.id.as_str().to_string());
        }
        for (doc_id, new_ids) in by_doc {
            let path = self.doc_chunks_path(&DocumentId::new(&doc_id));
            let mut existing: Vec<String> = self.read_json(&path)?.unwrap_or_default();
            for id in new_ids {
                if !existing.contains(&id) {
                    existing.push(id);
                }
            }
            self.write_json(&path, &existing)?;
        }
        Ok(())
    }

    async fn list_chunks_for_document(&self, id: &DocumentId) -> Result<Vec<ChunkId>> {
        let _g = self.lock.lock();
        let ids: Vec<String> = self
            .read_json(&self.doc_chunks_path(id))?
            .unwrap_or_default();
        Ok(ids.into_iter().map(ChunkId::new).collect())
    }

    async fn delete_chunk(&self, id: &ChunkId) -> Result<()> {
        let _g = self.lock.lock();
        let chunk: Option<Chunk> = self.read_json(&self.chunk_path(id))?;
        let Some(chunk) = chunk else {
            return Ok(());
        };
        let path = self.chunk_path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        let dcp = self.doc_chunks_path(&chunk.document_id);
        let mut existing: Vec<String> = self.read_json(&dcp)?.unwrap_or_default();
        existing.retain(|cid| cid != id.as_str());
        if existing.is_empty() {
            if dcp.exists() {
                std::fs::remove_file(dcp)?;
            }
        } else {
            self.write_json(&dcp, &existing)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn roundtrip_chunks_and_delete() {
        let dir = tempfile::tempdir().expect("temp");
        let store = FsDocStore::open(dir.path()).expect("open");
        let col = CollectionId::new("c");
        store
            .create_collection(&CollectionInfo {
                id: col.clone(),
                name: "c".into(),
                created_at: Utc::now(),
                chunk_count: 0,
            })
            .await
            .expect("col");

        let doc = Document {
            id: DocumentId::new("d1"),
            collection_id: col.clone(),
            text: "hello".into(),
            metadata: HashMap::new(),
        };
        store.upsert_documents(&[doc]).await.expect("doc");
        let chunk = Chunk {
            id: ChunkId::new("d1#0"),
            document_id: DocumentId::new("d1"),
            collection_id: col,
            text: "hello".into(),
            ordinal: 0,
            metadata: HashMap::new(),
        };
        store
            .put_chunks(std::slice::from_ref(&chunk))
            .await
            .expect("put");
        assert!(store.get_chunk(&chunk.id).await.unwrap().is_some());
        let listed = store
            .list_chunks_for_document(&DocumentId::new("d1"))
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        store
            .delete_document(&DocumentId::new("d1"))
            .await
            .expect("del");
        assert!(store.get_chunk(&chunk.id).await.unwrap().is_none());
    }
}
