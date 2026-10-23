//! Deterministic hash embedder (no ONNX). Useful for dense-path smoke tests.

use async_trait::async_trait;
use retrieva_core::traits::Embedder;
use retrieva_core::{Embedding, Result};

/// Fast, deterministic embedder with fixed dimensionality.
///
/// Not semantically meaningful — use [`super::OnnxEmbedder`] for retrieval quality.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dims: usize,
}

impl HashEmbedder {
    /// Create a hash embedder with the given output dimensionality.
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(1) }
    }

    fn embed_sync(&self, text: &str) -> Embedding {
        let mut values = vec![0.0_f32; self.dims];
        for (i, byte) in text.bytes().enumerate() {
            let idx = i % self.dims;
            values[idx] += f32::from(byte) / 255.0;
        }
        l2_normalize(&mut values);
        Embedding::new(values)
    }
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in values {
        *v /= norm;
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_and_unit_norm() {
        let emb = HashEmbedder::new(16);
        let a = emb.embed("hello").await.expect("a");
        let b = emb.embed("hello").await.expect("b");
        assert_eq!(a, b);
        let norm: f32 = a.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
