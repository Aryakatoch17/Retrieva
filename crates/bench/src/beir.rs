//! BEIR-format dataset loaders (`corpus.jsonl`, `queries.jsonl`, `qrels/*.tsv`).

use crate::metrics::Qrels;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// One corpus document.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusDoc {
    /// Document id (`_id` in BEIR JSONL).
    #[serde(rename = "_id")]
    pub id: String,
    /// Optional title.
    #[serde(default)]
    pub title: String,
    /// Body text.
    #[serde(default)]
    pub text: String,
}

impl CorpusDoc {
    /// Title + text joined for indexing.
    pub fn full_text(&self) -> String {
        if self.title.is_empty() {
            self.text.clone()
        } else if self.text.is_empty() {
            self.title.clone()
        } else {
            format!("{}\n\n{}", self.title, self.text)
        }
    }
}

/// One query.
#[derive(Debug, Clone, Deserialize)]
pub struct QueryDoc {
    /// Query id.
    #[serde(rename = "_id")]
    pub id: String,
    /// Query text.
    pub text: String,
}

/// Loaded BEIR-style split.
#[derive(Debug, Clone)]
pub struct BeirDataset {
    /// Root path.
    pub root: PathBuf,
    /// Corpus docs keyed by id.
    pub corpus: HashMap<String, CorpusDoc>,
    /// Queries keyed by id.
    pub queries: HashMap<String, QueryDoc>,
    /// Graded qrels.
    pub qrels: Qrels,
}

impl BeirDataset {
    /// Load from a directory containing `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`
    /// (or `qrels.tsv` at the root).
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let corpus_path =
            first_existing(&[root.join("corpus.jsonl"), root.join("corpus.jsonl.gz")])
                .with_context(|| format!("corpus.jsonl not found under {}", root.display()))?;
        if corpus_path.extension().is_some_and(|e| e == "gz") {
            bail!("gzipped corpus not supported yet; decompress first");
        }

        let queries_path =
            first_existing(&[root.join("queries.jsonl"), root.join("queries.jsonl.gz")])
                .with_context(|| format!("queries.jsonl not found under {}", root.display()))?;

        let qrels_path = first_existing(&[
            root.join("qrels").join("test.tsv"),
            root.join("qrels").join("dev.tsv"),
            root.join("qrels.tsv"),
        ])
        .with_context(|| format!("qrels tsv not found under {}", root.display()))?;

        let corpus = load_corpus(&corpus_path)?;
        let queries = load_queries(&queries_path)?;
        let qrels = load_qrels(&qrels_path)?;
        Ok(Self {
            root,
            corpus,
            queries,
            qrels,
        })
    }
}

fn first_existing(paths: &[PathBuf]) -> Option<PathBuf> {
    paths.iter().find(|p| p.exists()).cloned()
}

fn load_corpus(path: &Path) -> Result<HashMap<String, CorpusDoc>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = HashMap::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: CorpusDoc =
            serde_json::from_str(&line).with_context(|| format!("corpus line {}", i + 1))?;
        out.insert(doc.id.clone(), doc);
    }
    Ok(out)
}

fn load_queries(path: &Path) -> Result<HashMap<String, QueryDoc>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = HashMap::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let q: QueryDoc =
            serde_json::from_str(&line).with_context(|| format!("queries line {}", i + 1))?;
        out.insert(q.id.clone(), q);
    }
    Ok(out)
}

/// Load TREC/BEIR qrels: `query-id\tcorpus-id\tscore` (optional header).
pub fn load_qrels(path: &Path) -> Result<Qrels> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut qrels: Qrels = HashMap::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(['\t', ' ']).filter(|s| !s.is_empty()).collect();
        if parts.len() < 3 {
            continue;
        }
        // Skip header
        if i == 0 && parts[0].eq_ignore_ascii_case("query-id") {
            continue;
        }
        let qid = parts[0].to_string();
        let did = parts[1].to_string();
        let score: u32 = parts[2].parse().unwrap_or(1);
        qrels.entry(qid).or_default().insert(did, score);
    }
    Ok(qrels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_fixture_layout() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("corpus.jsonl"),
            r#"{"_id":"d1","title":"Rust","text":"systems language"}
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("queries.jsonl"),
            r#"{"_id":"q1","text":"rust language"}
"#,
        )
        .unwrap();
        std::fs::create_dir(root.join("qrels")).unwrap();
        let mut q = File::create(root.join("qrels").join("test.tsv")).unwrap();
        writeln!(q, "query-id\tcorpus-id\tscore").unwrap();
        writeln!(q, "q1\td1\t1").unwrap();

        let ds = BeirDataset::load(root).unwrap();
        assert_eq!(ds.corpus.len(), 1);
        assert_eq!(ds.queries.len(), 1);
        assert_eq!(ds.qrels["q1"]["d1"], 1);
    }
}
