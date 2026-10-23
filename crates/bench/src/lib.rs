//! BEIR eval harness, ranking metrics, and in-process load tests (Phase 9).

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod beir;
pub mod eval;
pub mod loadtest;
pub mod matrix;
pub mod metrics;
pub mod report;
pub mod rerank_ablate;

pub use beir::BeirDataset;
pub use eval::{parse_mode, run_beir_eval, BenchEmbedder, EvalOptions};
pub use loadtest::{host_fingerprint, run_beir_load_pair, run_load_test, LoadOptions, QueryMix};
pub use matrix::{
    assert_hybrid_beats_singles, assess_hybrid, run_dataset_matrix, write_matrix, MatrixOptions,
    MatrixReport, ModeRow,
};
pub use metrics::{mean_mrr, mean_ndcg_at_k, mean_recall_at_k};
pub use report::{compare_eval, write_eval, write_load, CompareReport, EvalReport, LoadReport};
pub use rerank_ablate::{run_ce_vs_maxsim_beir, write_rerank_ablate, RerankAblateReport};
