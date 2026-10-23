//! Append-only write-ahead log for incremental indexing.

use retrieva_core::{Chunk, CollectionId, DocumentId, Result, RetrievaError};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Durable op recorded before applying to indexes / doc store.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    /// Upsert chunks (and optional dense vectors aligned 1:1).
    UpsertChunks {
        /// Collection id.
        collection_id: CollectionId,
        /// Chunks to write.
        chunks: Vec<Chunk>,
        /// Dense vectors (same length as `chunks`), if present.
        #[serde(default)]
        vectors: Option<Vec<Vec<f32>>>,
    },
    /// Delete a document and all of its chunks.
    DeleteDocument {
        /// Collection id.
        collection_id: CollectionId,
        /// Document id.
        document_id: DocumentId,
    },
    /// Delete a single chunk.
    DeleteChunk {
        /// Collection id.
        collection_id: CollectionId,
        /// Chunk id.
        chunk_id: String,
    },
}

/// One sequenced WAL entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonic sequence number (starts at 1).
    pub seq: u64,
    /// Operation payload.
    pub op: WalOp,
}

/// Checkpoint of the last fully applied sequence.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WalCheckpoint {
    /// Last applied seq (0 = nothing applied).
    pub applied_seq: u64,
}

/// File-backed WAL under `data_dir/wal/`.
pub struct WriteAheadLog {
    dir: PathBuf,
    log_path: PathBuf,
    checkpoint_path: PathBuf,
    next_seq: u64,
}

impl WriteAheadLog {
    /// Open or create the WAL directory, loading checkpoint + next seq.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = data_dir.as_ref().join("wal");
        std::fs::create_dir_all(&dir)?;
        let log_path = dir.join("log.jsonl");
        let checkpoint_path = dir.join("checkpoint.json");
        let checkpoint = Self::load_checkpoint(&checkpoint_path)?;
        let max_seq = Self::scan_max_seq(&log_path)?;
        let next_seq = max_seq.max(checkpoint.applied_seq) + 1;
        Ok(Self {
            dir,
            log_path,
            checkpoint_path,
            next_seq,
        })
    }

    /// Append an operation (fsync'd) and return its sequence number.
    pub fn append(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = WalEntry { seq, op };
        let line = serde_json::to_string(&entry)
            .map_err(|e| RetrievaError::Storage(format!("wal encode: {e}")))?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        writeln!(file, "{line}")?;
        file.sync_all()?;
        debug!(seq, "wal appended");
        Ok(seq)
    }

    /// Persist that all ops through `seq` have been applied to durable indexes.
    pub fn checkpoint(&self, seq: u64) -> Result<()> {
        let cp = WalCheckpoint { applied_seq: seq };
        let tmp = self.dir.join("checkpoint.json.tmp");
        let json = serde_json::to_string_pretty(&cp)
            .map_err(|e| RetrievaError::Storage(format!("checkpoint encode: {e}")))?;
        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            file.write_all(json.as_bytes())?;
            file.sync_all()?;
        } // drop handle before rename (required on Windows)
        std::fs::rename(&tmp, &self.checkpoint_path)?;
        info!(applied_seq = seq, "wal checkpoint");
        Ok(())
    }

    /// Current checkpoint.
    pub fn applied_seq(&self) -> Result<u64> {
        Ok(Self::load_checkpoint(&self.checkpoint_path)?.applied_seq)
    }

    /// Entries with `seq > after_seq`, in order.
    pub fn entries_after(&self, after_seq: u64) -> Result<Vec<WalEntry>> {
        if !self.log_path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.log_path)?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: WalEntry = serde_json::from_str(&line)
                .map_err(|e| RetrievaError::Storage(format!("wal decode: {e}")))?;
            if entry.seq > after_seq {
                out.push(entry);
            }
        }
        out.sort_by_key(|e| e.seq);
        Ok(out)
    }

    fn load_checkpoint(path: &Path) -> Result<WalCheckpoint> {
        if !path.exists() {
            return Ok(WalCheckpoint::default());
        }
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| RetrievaError::Storage(format!("checkpoint decode: {e}")))
    }

    fn scan_max_seq(log_path: &Path) -> Result<u64> {
        if !log_path.exists() {
            return Ok(0);
        }
        let file = File::open(log_path)?;
        let reader = BufReader::new(file);
        let mut max = 0u64;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: WalEntry = serde_json::from_str(&line)
                .map_err(|e| RetrievaError::Storage(format!("wal decode: {e}")))?;
            max = max.max(entry.seq);
        }
        Ok(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_replay() {
        let dir = tempfile::tempdir().expect("temp");
        let mut wal = WriteAheadLog::open(dir.path()).expect("open");
        let seq1 = wal
            .append(WalOp::DeleteChunk {
                collection_id: CollectionId::new("c"),
                chunk_id: "x".into(),
            })
            .expect("a1");
        assert_eq!(seq1, 1);
        wal.checkpoint(1).expect("cp");
        let seq2 = wal
            .append(WalOp::DeleteDocument {
                collection_id: CollectionId::new("c"),
                document_id: DocumentId::new("d"),
            })
            .expect("a2");
        assert_eq!(seq2, 2);

        let wal2 = WriteAheadLog::open(dir.path()).expect("reopen");
        assert_eq!(wal2.applied_seq().unwrap(), 1);
        let pending = wal2.entries_after(1).expect("pending");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].seq, 2);
    }
}
