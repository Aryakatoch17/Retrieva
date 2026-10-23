//! End-to-end ingest: walk paths → parse → chunk → [`Document`] / [`Chunk`].

use crate::chunk::chunk_text;
use crate::parse::{extract_text, is_supported};
use retrieva_core::{
    Chunk, ChunkId, CollectionId, Document, DocumentId, IngestConfig, RetrievaError,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};
use walkdir::WalkDir;

/// Result of ingesting one or more source files.
#[derive(Debug, Clone, Default)]
pub struct IngestBatch {
    /// Source documents (full text).
    pub documents: Vec<Document>,
    /// Chunks derived from those documents.
    pub chunks: Vec<Chunk>,
    /// Files successfully parsed.
    pub files_ok: usize,
    /// Files skipped (unsupported / empty / errors).
    pub files_skipped: usize,
}

/// Walk `path` (file or directory), parse supported documents, and chunk them.
pub fn ingest_path(
    path: &Path,
    collection_id: &CollectionId,
    ingest: &IngestConfig,
) -> Result<IngestBatch, RetrievaError> {
    if !path.exists() {
        return Err(RetrievaError::Ingest(format!(
            "path does not exist: {}",
            path.display()
        )));
    }

    let files = collect_files(path)?;
    let mut batch = IngestBatch::default();

    for file in files {
        match ingest_file(&file, path, collection_id, ingest) {
            Ok(Some((doc, chunks))) => {
                batch.documents.push(doc);
                batch.chunks.extend(chunks);
                batch.files_ok += 1;
            }
            Ok(None) => {
                batch.files_skipped += 1;
            }
            Err(e) => {
                warn!(path = %file.display(), error = %e, "skipping file");
                batch.files_skipped += 1;
            }
        }
    }

    Ok(batch)
}

/// Parse and chunk a single file. Returns `Ok(None)` when the file is empty.
pub fn ingest_file(
    file: &Path,
    root: &Path,
    collection_id: &CollectionId,
    ingest: &IngestConfig,
) -> Result<Option<(Document, Vec<Chunk>)>, RetrievaError> {
    if !is_supported(file) {
        return Ok(None);
    }

    let text = extract_text(file)?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        debug!(path = %file.display(), "empty document, skipping");
        return Ok(None);
    }

    let doc_id = document_id_for(file, root);
    let mut metadata = HashMap::new();
    metadata.insert("source_path".into(), file.display().to_string());

    let doc = Document {
        id: DocumentId::new(doc_id.clone()),
        collection_id: collection_id.clone(),
        text: trimmed.to_string(),
        metadata: metadata.clone(),
    };

    let parts = chunk_text(trimmed, ingest.chunk_size, ingest.chunk_overlap);
    let chunks: Vec<Chunk> = parts
        .into_iter()
        .enumerate()
        .map(|(ordinal, text)| Chunk {
            id: ChunkId::new(format!("{doc_id}#{ordinal}")),
            document_id: DocumentId::new(doc_id.clone()),
            collection_id: collection_id.clone(),
            text,
            ordinal: ordinal as u32,
            metadata: metadata.clone(),
        })
        .collect();

    Ok(Some((doc, chunks)))
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>, RetrievaError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(path).follow_links(false) {
        let entry = entry.map_err(|e| {
            RetrievaError::Ingest(format!("walkdir error under {}: {e}", path.display()))
        })?;
        if entry.file_type().is_file() && is_supported(entry.path()) {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn document_id_for(file: &Path, root: &Path) -> String {
    let relative = file.strip_prefix(root).unwrap_or(file);
    let raw = relative.to_string_lossy().replace('\\', "/");
    // Stable, filesystem-safe id derived from relative path.
    raw.trim_start_matches("./").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn ingests_directory_of_text_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.txt");
        let b = dir.path().join("sub");
        fs::create_dir_all(&b).expect("mkdir");
        let b_file = b.join("b.md");
        write_file(&a, "alpha beta gamma delta epsilon zeta");
        write_file(&b_file, "one two three four five six seven");

        let cfg = IngestConfig {
            chunk_size: 4,
            chunk_overlap: 1,
        };
        let batch = ingest_path(dir.path(), &CollectionId::new("demo"), &cfg).expect("ingest");
        assert_eq!(batch.files_ok, 2);
        assert!(batch.chunks.len() >= 2);
        assert!(batch
            .chunks
            .iter()
            .all(|c| c.collection_id.as_str() == "demo"));
    }

    fn write_file(path: &Path, body: &str) {
        let mut f = fs::File::create(path).expect("create");
        writeln!(f, "{body}").expect("write");
    }
}
