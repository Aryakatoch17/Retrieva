//! Tantivy-backed BM25 [`SparseIndex`].

use async_trait::async_trait;
use parking_lot::Mutex;
use retrieva_core::traits::SparseIndex;
use retrieva_core::{
    Chunk, ChunkId, CollectionId, Result, RetrievaError, ScoreBreakdown, ScoredChunk,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TantivyDocument, TextFieldIndexing, TextOptions, Value,
    INDEXED, STORED, STRING,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};
use tracing::debug;

const WRITER_HEAP_BYTES: usize = 50_000_000;

/// Per-collection Tantivy handles.
struct CollectionIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    fields: SchemaFields,
}

#[derive(Clone, Copy)]
struct SchemaFields {
    chunk_id: Field,
    document_id: Field,
    collection_id: Field,
    text: Field,
    ordinal: Field,
    metadata_json: Field,
}

/// On-disk Tantivy BM25 index rooted at `data_dir/indexes/<collection>/tantivy`.
///
/// One directory per collection keeps deletes and future WAL work isolated.
pub struct TantivySparseIndex {
    root: PathBuf,
    open: Mutex<HashMap<String, Arc<Mutex<CollectionIndex>>>>,
}

impl TantivySparseIndex {
    /// Open (or create) the sparse index under `data_dir`.
    pub fn open(data_dir: impl Into<PathBuf>) -> Result<Self> {
        let root = data_dir.into().join("indexes");
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            open: Mutex::new(HashMap::new()),
        })
    }

    /// Remove all chunks for a document within a collection (single commit).
    pub async fn remove_document(
        &self,
        collection_id: &CollectionId,
        document_id: &retrieva_core::DocumentId,
    ) -> Result<()> {
        let handle = self.get_or_open(collection_id.as_str())?;
        let mut guard = handle.lock();
        let term = Term::from_field_text(guard.fields.document_id, document_id.as_str());
        guard.writer.delete_term(term);
        guard
            .writer
            .commit()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        guard
            .reader
            .reload()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        Ok(())
    }

    /// Remove one chunk within a known collection (single commit).
    pub async fn remove_in_collection(
        &self,
        collection_id: &CollectionId,
        chunk_id: &ChunkId,
    ) -> Result<()> {
        let handle = self.get_or_open(collection_id.as_str())?;
        let mut guard = handle.lock();
        let term = Term::from_field_text(guard.fields.chunk_id, chunk_id.as_str());
        guard.writer.delete_term(term);
        guard
            .writer
            .commit()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        guard
            .reader
            .reload()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        Ok(())
    }

    /// Index many chunks with a single commit per collection (faster ingest).
    pub async fn upsert_batch(&self, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let mut by_collection: HashMap<String, Vec<&Chunk>> = HashMap::new();
        for chunk in chunks {
            by_collection
                .entry(chunk.collection_id.as_str().to_string())
                .or_default()
                .push(chunk);
        }

        for (collection, group) in by_collection {
            let n = group.len();
            let handle = self.get_or_open(&collection)?;
            let mut guard = handle.lock();
            for chunk in &group {
                Self::upsert_sync(&mut guard, chunk)?;
            }
            guard
                .writer
                .commit()
                .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
            guard
                .reader
                .reload()
                .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
            debug!(collection = %collection, n, "committed BM25 batch");
        }
        Ok(())
    }

    fn collection_dir(&self, collection_id: &str) -> PathBuf {
        self.root.join(collection_id).join("tantivy")
    }

    fn get_or_open(&self, collection_id: &str) -> Result<Arc<Mutex<CollectionIndex>>> {
        {
            let guard = self.open.lock();
            if let Some(existing) = guard.get(collection_id) {
                return Ok(Arc::clone(existing));
            }
        }

        let dir = self.collection_dir(collection_id);
        std::fs::create_dir_all(&dir)?;
        let opened = open_collection(&dir)?;
        let arc = Arc::new(Mutex::new(opened));

        let mut guard = self.open.lock();
        Ok(Arc::clone(
            guard.entry(collection_id.to_string()).or_insert(arc),
        ))
    }

    fn upsert_sync(col: &mut CollectionIndex, chunk: &Chunk) -> Result<()> {
        let f = col.fields;
        let id_term = Term::from_field_text(f.chunk_id, chunk.id.as_str());
        col.writer.delete_term(id_term);

        let meta = serde_json::to_string(&chunk.metadata)
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;

        col.writer
            .add_document(doc!(
                f.chunk_id => chunk.id.as_str(),
                f.document_id => chunk.document_id.as_str(),
                f.collection_id => chunk.collection_id.as_str(),
                f.text => chunk.text.as_str(),
                f.ordinal => u64::from(chunk.ordinal),
                f.metadata_json => meta.as_str(),
            ))
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        Ok(())
    }

    fn search_sync(
        col: &CollectionIndex,
        collection_id: &CollectionId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        let searcher = col.reader.searcher();
        let f = col.fields;

        let parser = QueryParser::for_index(&col.index, vec![f.text]);
        // BEIR / user queries often contain query-syntax chars (`'`, `(`, `:`, …).
        // Sanitize to plain terms so QueryParser never fails on real corpora.
        let safe = sanitize_bm25_query(query);
        if safe.is_empty() {
            return Ok(Vec::new());
        }
        let text_query = parser
            .parse_query(&safe)
            .map_err(|e| RetrievaError::SparseIndex(format!("query parse error: {e}")))?;

        let collection_term = Term::from_field_text(f.collection_id, collection_id.as_str());
        let collection_query = TermQuery::new(collection_term, IndexRecordOption::Basic);

        let combined: BooleanQuery = BooleanQuery::new(vec![
            (Occur::Must, Box::new(text_query) as _),
            (Occur::Must, Box::new(collection_query) as _),
        ]);

        let top_docs = searcher
            .search(&combined, &TopDocs::with_limit(top_k).order_by_score())
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let retrieved: TantivyDocument = searcher
                .doc(addr)
                .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
            let chunk = document_to_chunk(&retrieved, f)?;
            hits.push(ScoredChunk {
                chunk,
                scores: ScoreBreakdown {
                    bm25: Some(score),
                    final_score: score,
                    ..ScoreBreakdown::default()
                },
            });
        }
        Ok(hits)
    }
}

/// Keep letters/digits (any script) and plain whitespace; drop query-syntax and fancy quotes.
fn sanitize_bm25_query(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    for c in query.chars() {
        if c.is_alphanumeric() || c.is_whitespace() {
            out.push(c);
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn build_schema() -> (Schema, SchemaFields) {
    let mut builder = Schema::builder();

    let chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
    let document_id = builder.add_text_field("document_id", STRING | STORED);
    // Indexed as STRING so we can filter with TermQuery; also stored.
    let collection_id = builder.add_text_field("collection_id", STRING | STORED);

    let text_opts = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let text = builder.add_text_field("text", text_opts);

    let ordinal = builder.add_u64_field("ordinal", STORED | INDEXED);
    let metadata_json =
        builder.add_text_field("metadata_json", TextOptions::default().set_stored());

    let schema = builder.build();
    let fields = SchemaFields {
        chunk_id,
        document_id,
        collection_id,
        text,
        ordinal,
        metadata_json,
    };
    (schema, fields)
}

fn open_collection(dir: &Path) -> Result<CollectionIndex> {
    let (schema, fields) = build_schema();
    let index = Index::open_or_create(
        tantivy::directory::MmapDirectory::open(dir)
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?,
        schema,
    )
    .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;

    let writer = index
        .writer(WRITER_HEAP_BYTES)
        .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;

    Ok(CollectionIndex {
        index,
        writer,
        reader,
        fields,
    })
}

fn document_to_chunk(doc: &TantivyDocument, f: SchemaFields) -> Result<Chunk> {
    let chunk_id = require_str(doc, f.chunk_id, "chunk_id")?;
    let document_id = require_str(doc, f.document_id, "document_id")?;
    let collection_id = require_str(doc, f.collection_id, "collection_id")?;
    let text = require_str(doc, f.text, "text")?;
    let ordinal = doc
        .get_first(f.ordinal)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RetrievaError::SparseIndex("missing ordinal".into()))?;
    let meta_raw = doc
        .get_first(f.metadata_json)
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    let metadata: HashMap<String, String> = serde_json::from_str(meta_raw).unwrap_or_default();

    Ok(Chunk {
        id: ChunkId::new(chunk_id),
        document_id: retrieva_core::DocumentId::new(document_id),
        collection_id: CollectionId::new(collection_id),
        text,
        ordinal: ordinal as u32,
        metadata,
    })
}

fn require_str(doc: &TantivyDocument, field: Field, name: &str) -> Result<String> {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RetrievaError::SparseIndex(format!("missing stored field '{name}'")))
}

#[async_trait]
impl SparseIndex for TantivySparseIndex {
    async fn upsert(&self, chunk: &Chunk) -> Result<()> {
        let handle = self.get_or_open(chunk.collection_id.as_str())?;
        let mut guard = handle.lock();
        Self::upsert_sync(&mut guard, chunk)?;
        guard
            .writer
            .commit()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        guard
            .reader
            .reload()
            .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        Ok(())
    }

    async fn remove(&self, chunk_id: &ChunkId) -> Result<()> {
        // Chunk ids are globally unique (`doc#ordinal`); scan open collections,
        // and also try to delete from any on-disk collection dir.
        let collection_ids: Vec<String> = {
            let guard = self.open.lock();
            guard.keys().cloned().collect()
        };

        let mut dirs: Vec<String> = collection_ids;
        if self.root.exists() {
            for entry in std::fs::read_dir(&self.root)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !dirs.contains(&name) {
                        dirs.push(name);
                    }
                }
            }
        }

        for collection in dirs {
            let handle = self.get_or_open(&collection)?;
            let mut guard = handle.lock();
            let term = Term::from_field_text(guard.fields.chunk_id, chunk_id.as_str());
            guard.writer.delete_term(term);
            guard
                .writer
                .commit()
                .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
            guard
                .reader
                .reload()
                .map_err(|e| RetrievaError::SparseIndex(e.to_string()))?;
        }
        Ok(())
    }

    async fn search(
        &self,
        collection_id: &CollectionId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>> {
        if query.trim().is_empty() {
            return Err(RetrievaError::InvalidArgument(
                "query must not be empty".into(),
            ));
        }
        if top_k == 0 {
            return Err(RetrievaError::InvalidArgument("top_k must be >= 1".into()));
        }

        let dir = self.collection_dir(collection_id.as_str());
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let handle = self.get_or_open(collection_id.as_str())?;
        let guard = handle.lock();
        Self::search_sync(&guard, collection_id, query, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieva_core::DocumentId;
    use std::collections::HashMap;

    fn sample_chunk(collection: &str, id: &str, text: &str) -> Chunk {
        Chunk {
            id: ChunkId::new(id),
            document_id: DocumentId::new("doc1"),
            collection_id: CollectionId::new(collection),
            text: text.into(),
            ordinal: 0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn sanitize_strips_query_syntax() {
        let s = sanitize_bm25_query("virus' (ComYMV) genome: 7489");
        assert_eq!(s, "virus ComYMV genome 7489");
        let fancy = sanitize_bm25_query("What does “IPOs” mean with an `underwriting discount`?");
        assert_eq!(fancy, "What does IPOs mean with an underwriting discount");
    }

    #[tokio::test]
    async fn bm25_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = TantivySparseIndex::open(dir.path()).expect("open");

        index
            .upsert_batch(&[
                sample_chunk("col", "c1", "rust retrieval engine with bm25"),
                sample_chunk("col", "c2", "python web framework flask django"),
            ])
            .await
            .expect("upsert");

        let hits = index
            .search(&CollectionId::new("col"), "rust bm25", 5)
            .await
            .expect("search");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk.id.as_str(), "c1");
        assert!(hits[0].scores.bm25.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn collection_isolation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = TantivySparseIndex::open(dir.path()).expect("open");
        index
            .upsert(&sample_chunk("a", "c1", "shared keyword alpha"))
            .await
            .expect("upsert a");
        index
            .upsert(&sample_chunk("b", "c2", "shared keyword beta"))
            .await
            .expect("upsert b");

        let hits = index
            .search(&CollectionId::new("a"), "keyword", 10)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.collection_id.as_str(), "a");
    }
}
