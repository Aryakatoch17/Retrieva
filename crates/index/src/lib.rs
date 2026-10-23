//! Index backends: Tantivy BM25, usearch ANN, FsDocStore + WAL (Phase 6).
//!
//! Phase 2: [`TantivySparseIndex`] + [`Bm25Retriever`].
//! Phase 3: [`UsearchVectorIndex`] + [`DenseRetriever`].
//! Phase 6: [`WriteAheadLog`], [`FsDocStore`], [`IncrementalIndex`].

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod incremental;
pub mod keys;
pub mod retriever;
pub mod sparse;
pub mod store;
pub mod vector;
pub mod wal;

pub use incremental::IncrementalIndex;
pub use keys::chunk_key;
pub use retriever::{Bm25Retriever, DenseRetriever};
pub use sparse::TantivySparseIndex;
pub use store::FsDocStore;
pub use vector::{CollectionVectorIndex, UsearchVectorIndex};
pub use wal::{WalCheckpoint, WalEntry, WalOp, WriteAheadLog};
